"""
Can an LLM replace training data?

Phases (run one at a time):
  baselines  Phase 3: train networks on 5..250 labeled examples (20 repeats) + random weights
  pilot      Phase 4: ONE Codex call (tumor, simple network) to check the plumbing
  llm        Phase 5: all LLM conditions, fresh no-tools Codex sessions
  combine    Phase 6: LLM weights as a starting point, then a little training
  report     Phase 7: tables + charts

Usage (from the project root):
  .venv/bin/python experiments/knowledge_vs_data/run.py baselines
  .venv/bin/python experiments/knowledge_vs_data/run.py pilot
  .venv/bin/python experiments/knowledge_vs_data/run.py llm --repeats 5
  .venv/bin/python experiments/knowledge_vs_data/run.py combine
  .venv/bin/python experiments/knowledge_vs_data/run.py report
"""
import argparse, json, re, statistics, subprocess, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from tasks import get_tasks                     # noqa: E402


def _call_codex(*a, **k):
    """Shared Codex wrapper from experiments/add_digit9/run.py, loaded under its own name."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("add_digit9_run", HERE.parent / "add_digit9" / "run.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.call_codex(*a, **k)

torch.set_num_threads(4)
SIZES = [5, 10, 25, 50, 100, 250]
REPEATS = 20
HIDDEN = 8
STEPS, LR, L2 = 300, 0.05, 1e-3


# ---------------------------------------------------------------- networks
def forward(params, Z):
    Z = torch.as_tensor(Z, dtype=torch.float32)
    if "w" in params:                                           # simple network
        return Z @ params["w"] + params["b"]
    h = F.relu(Z @ params["W1"].T + params["b1"])               # small network
    return h @ params["w2"] + params["b2"]


def init_params(kind, d, gen):
    if kind == "simple":
        return {"w": torch.randn(d, generator=gen) * (1 / d) ** 0.5, "b": torch.zeros(())}
    return {"W1": torch.randn(HIDDEN, d, generator=gen) * (2 / d) ** 0.5, "b1": torch.zeros(HIDDEN),
            "w2": torch.randn(HIDDEN, generator=gen) * (1 / HIDDEN) ** 0.5, "b2": torch.zeros(())}


def train(params, Z, y):
    p = {k: v.clone().float().requires_grad_(True) for k, v in params.items()}
    opt = torch.optim.Adam(p.values(), lr=LR)
    yt = torch.as_tensor(y, dtype=torch.float32)
    for _ in range(STEPS):
        loss = F.binary_cross_entropy_with_logits(forward(p, Z), yt)
        loss = loss + L2 * sum((v ** 2).sum() for k, v in p.items() if k in ("w", "W1", "w2"))
        opt.zero_grad(); loss.backward(); opt.step()
    return {k: v.detach() for k, v in p.items()}


def evaluate(params, Z, y):
    with torch.no_grad():
        s = forward(params, Z).numpy()
    acc = float(((s > 0).astype(int) == y).mean() * 100)
    try:
        auc = float(roc_auc_score(y, s))
    except ValueError:
        auc = float("nan")
    if not np.isfinite(s).all():
        return float("nan"), float("nan")
    return acc, auc


def sample_idx(y, n, rng):
    while True:                                                  # need both classes present
        idx = rng.choice(len(y), size=n, replace=False)
        if 0 < y[idx].sum() < n:
            return idx


# ---------------------------------------------------------------- Phase 3
def phase_baselines(args):
    out = {}
    for t in get_tasks():
        d = t["Zp"].shape[1]
        out[t["name"]] = {}
        for kind in ["simple", "small"]:
            rows = {}
            gen = torch.Generator().manual_seed(0)
            rnd = [evaluate(init_params(kind, d, gen), t["Zt"], t["yt"]) for _ in range(REPEATS)]
            rows["random weights"] = rnd
            for n in SIZES:
                rng = np.random.default_rng(1000 + n)
                res = []
                for _ in range(REPEATS):
                    idx = sample_idx(t["yp"], n, rng)
                    p = train(init_params(kind, d, gen), t["Zp"][idx], t["yp"][idx])
                    res.append(evaluate(p, t["Zt"], t["yt"]))
                rows[f"train n={n}"] = res
            rows["train all pool"] = [evaluate(train(init_params(kind, d, gen), t["Zp"], t["yp"]),
                                               t["Zt"], t["yt"])]
            out[t["name"]][kind] = rows
            print(f"\n{t['name']} / {kind} network        accuracy           AUROC")
            for name, res in rows.items():
                a = [r[0] for r in res]; u = [r[1] for r in res]
                print(f"  {name:16s} {statistics.mean(a):5.1f}% (sd {statistics.pstdev(a):4.1f})"
                      f"   {statistics.mean(u):.3f}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baselines.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {RESULTS / 'baselines.json'}")


# ---------------------------------------------------------------- prompts
NET_SPEC = {
    "simple": ("NETWORK: score = w . z + b ; probability = sigmoid(score).\n"
               "Output ONLY a JSON object, then REASONS: (max 120 words):\n"
               '{{"w": [<{d} numbers, one per feature, in the order listed>], "b": <number>}}'),
    "small": ("NETWORK: h = relu(W1 z + b1) with {h} hidden neurons; score = w2 . h + b2 ; "
              "probability = sigmoid(score).\n"
              "Output ONLY a JSON object, then REASONS: (max 120 words):\n"
              '{{"W1": [<{h} lists of {d} numbers>], "b1": [<{h} numbers>], '
              '"w2": [<{h} numbers>], "b2": <number>}}'),
}


def build_prompt(t, condition):
    d = len(t["names"])
    head = ("This is a research experiment; failure is a valid result.\n"
            "Do NOT run any code or use any tools. Work it out by reasoning and answer directly.\n"
            "You get NO training data. Use only your own knowledge.\n\n")
    if condition == "hidden_names":
        lines = [head, f"Each case has {d} numeric measurements, called x1..x{d}. "
                 "Nothing else is known about them.",
                 t["question"],
                 f"Inputs z1..z{d} are the measurements standardized to mean 0, std 1.", ""]
        lines.append(NET_SPEC["simple"].format(d=d, h=HIDDEN))
        return "\n".join(lines) + "\n"
    feat = "\n".join(f"{i + 1}. {n}  (mean {m:.4g}, std {s:.4g})"
                     for i, (n, m, s) in enumerate(zip(t["names"], t["mu"], t["sd"])))
    lines = [head, t["context"], ""]
    if t["rule"]:
        lines += ["RULE: " + t["rule"], ""]
    lines += [t["question"], "", "FEATURES (in order), with their average and spread:", feat, ""]
    if condition == "python_rule":
        lines += ["Write a Python function predict(f) where f is a dict mapping each feature name "
                  "above to its RAW value (not standardized). Return the probability (0..1) of the "
                  "positive outcome. Use only plain Python and the math module.",
                  "Output ONLY one ```python code block, then REASONS: (max 120 words)."]
    else:
        kind = "simple" if condition == "weights_simple" else "small"
        lines += [f"Inputs: z_i = (raw_i - mean_i) / std_i for the {d} features above.",
                  NET_SPEC[kind].format(d=d, h=HIDDEN)]
    return "\n".join(lines) + "\n"


def parse_json_params(txt, kind, d):
    txt = re.split(r"REASONS", txt, flags=re.I)[0]
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        if kind == "simple":
            w = torch.tensor(obj["w"], dtype=torch.float32); b = torch.tensor(float(obj["b"]))
            return {"w": w, "b": b} if w.shape == (d,) else None
        W1 = torch.tensor(obj["W1"], dtype=torch.float32)
        b1 = torch.tensor(obj["b1"], dtype=torch.float32)
        w2 = torch.tensor(obj["w2"], dtype=torch.float32)
        b2 = torch.tensor(float(obj["b2"]))
        ok = W1.shape == (HIDDEN, d) and b1.shape == (HIDDEN,) and w2.shape == (HIDDEN,)
        return {"W1": W1, "b1": b1, "w2": w2, "b2": b2} if ok else None
    except (KeyError, TypeError, ValueError):
        return None


RUNNER = r"""
import json, math, sys
ns = {"math": math}
exec(open(sys.argv[1]).read(), ns)
rows = json.load(open(sys.argv[2]))
out = []
for r in rows:
    try: out.append(float(ns["predict"](r)))
    except Exception: out.append(float("nan"))
print(json.dumps(out))
"""


def score_python_rule(txt, t):
    m = re.search(r"```python\s*(.*?)```", txt, re.S)
    if not m or "def predict" not in m.group(1):
        return None
    tmp = Path(tempfile.mkdtemp(prefix="kvd_rule_"))
    (tmp / "rule.py").write_text(m.group(1))
    (tmp / "runner.py").write_text(RUNNER)
    rows = [dict(zip(t["names"], map(float, x))) for x in t["Xt"]]
    (tmp / "rows.json").write_text(json.dumps(rows))
    try:
        p = subprocess.run([sys.executable, "-I", "runner.py", "rule.py", "rows.json"], cwd=tmp,
                           capture_output=True, text=True, timeout=60)
        probs = np.array(json.loads(p.stdout))
    except Exception:
        return None
    if not np.isfinite(probs).all():
        return None
    acc = float(((probs > 0.5).astype(int) == t["yt"]).mean() * 100)
    try:
        auc = float(roc_auc_score(t["yt"], probs))
    except ValueError:
        auc = float("nan")
    return acc, auc


CONDITIONS = ["weights_simple", "weights_small", "python_rule", "hidden_names"]


def run_llm(args, jobs, out_dir):
    tasks = {t["name"]: t for t in get_tasks()}
    stop = threading.Event()

    def one(job):
        tname, cond, r = job
        t = tasks[tname]; d = len(t["names"])
        prompt = build_prompt(t, cond)
        (out_dir / f"prompt_{tname}_{cond}.txt").write_text(prompt)
        rec = {"task": tname, "condition": cond, "repeat": r, "valid": False}
        if stop.is_set():
            rec["skipped"] = "usage limit"; return rec
        res = _call_codex(prompt, out_dir / "runs" / f"{tname}_{cond}_r{r}", args.timeout)
        rec.update(seconds=res["seconds"], tools_used=len(res["tools"]), tools=res["tools"])
        if res["quota"]:
            stop.set(); rec["skipped"] = "usage limit"; return rec
        if res["tools"]:
            rec["invalid_reason"] = "used tools"; return rec
        if cond == "python_rule":
            sc = score_python_rule(res["reply"], t)
        else:
            kind = "small" if cond == "weights_small" else "simple"
            params = parse_json_params(res["reply"], kind, d)
            sc = evaluate(params, t["Zt"], t["yt"]) if params else None
            if params:
                torch.save(params, out_dir / "runs" / f"{tname}_{cond}_r{r}" / "params.pt")
        if sc is None or not np.isfinite(sc[0]):
            rec["invalid_reason"] = "unparseable"; return rec
        rec.update(valid=True, acc=sc[0], auc=sc[1])
        return rec

    results = []
    with ThreadPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            rec = f.result(); results.append(rec)
            msg = (f"acc {rec['acc']:5.1f}%  AUROC {rec['auc']:.3f}" if rec["valid"]
                   else rec.get("skipped") or rec.get("invalid_reason"))
            print(f"  {rec['task']:5s} {rec['condition']:14s} r{rec['repeat']}: {msg}"
                  f"  ({rec.get('seconds', 0)}s, tools={rec.get('tools_used', 0)})", flush=True)
            (out_dir / "llm_results.json").write_text(json.dumps(results, indent=1))
    if stop.is_set():
        print("\n!! Codex usage limit reached: remaining sessions were skipped, not counted.")
    return results


def phase_pilot(args):
    out_dir = RESULTS / ("pilot-" + datetime.now().strftime("%Y%m%d-%H%M%S")); out_dir.mkdir(parents=True)
    res = run_llm(args, [("tumor", "weights_simple", 0)], out_dir)
    print("\nreply:\n" + (out_dir / "runs" / "tumor_weights_simple_r0" / "reply.txt").read_text()[:3000])
    print(f"\nsaved to {out_dir}")


def phase_llm(args):
    out_dir = RESULTS / ("llm-" + datetime.now().strftime("%Y%m%d-%H%M%S")); out_dir.mkdir(parents=True)
    conds = args.conditions.split(",")
    jobs = [(t, c, r) for r in range(args.repeats) for t in ["tumor", "loan"] for c in conds]
    print(f"running {len(jobs)} Codex sessions, {args.workers} at a time ...", flush=True)
    run_llm(args, jobs, out_dir)
    print(f"\nsaved to {out_dir}")


# ---------------------------------------------------------------- Phase 6
def latest(prefix):
    runs = sorted(RESULTS.glob(prefix + "-*"))
    return runs[-1] if runs else None


def phase_combine(args):
    llm_dir = Path(args.run) if args.run else latest("llm")
    tasks = {t["name"]: t for t in get_tasks()}
    out = {}
    for tname, t in tasks.items():
        d = t["Zp"].shape[1]
        for cond, kind in [("weights_simple", "simple"), ("weights_small", "small")]:
            inits = [torch.load(p) for p in sorted((llm_dir / "runs").glob(f"{tname}_{cond}_r*/params.pt"))]
            if not inits:
                continue
            key = f"{tname}/{kind}"; out[key] = {}
            print(f"\n{key}: {len(inits)} LLM starting points     from LLM      from scratch")
            for n in [5, 10, 25, 50]:
                rng = np.random.default_rng(2000 + n); gen = torch.Generator().manual_seed(n)
                llm_res, scr_res = [], []
                for _ in range(REPEATS):
                    idx = sample_idx(t["yp"], n, rng)
                    for p0 in inits:
                        llm_res.append(evaluate(train(p0, t["Zp"][idx], t["yp"][idx]), t["Zt"], t["yt"]))
                    scr_res.append(evaluate(train(init_params(kind, d, gen), t["Zp"][idx], t["yp"][idx]),
                                            t["Zt"], t["yt"]))
                out[key][n] = {"from_llm": llm_res, "from_scratch": scr_res}
                ma = statistics.mean(r[0] for r in llm_res); mb = statistics.mean(r[0] for r in scr_res)
                print(f"  n={n:<3d}                                {ma:5.1f}%       {mb:5.1f}%")
    (llm_dir / "combine.json").write_text(json.dumps(out, indent=1))
    print(f"\nsaved {llm_dir / 'combine.json'}")


# ---------------------------------------------------------------- Phase 7
def crossover(llm_acc, curve):
    """Smallest number of examples at which average training matches the LLM (linear interp)."""
    pts = [(n, statistics.mean(r[0] for r in curve[f"train n={n}"])) for n in SIZES]
    if llm_acc <= pts[0][1]:
        return f"fewer than {pts[0][0]}"
    for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
        if a0 < llm_acc <= a1:
            return f"about {round(n0 + (llm_acc - a0) / (a1 - a0) * (n1 - n0))}"
    return f"more than {pts[-1][0]}"


def phase_report(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    llm_dir = Path(args.run) if args.run else latest("llm")
    base = json.loads((RESULTS / "baselines.json").read_text())
    llm = json.loads((llm_dir / "llm_results.json").read_text())
    comb = json.loads((llm_dir / "combine.json").read_text()) if (llm_dir / "combine.json").exists() else {}
    label = {"weights_simple": "LLM weights, simple net", "weights_small": "LLM weights, small net",
             "python_rule": "LLM Python rule", "hidden_names": "LLM weights, names hidden"}
    md = [f"# Can an LLM replace training data? ({llm_dir.name})", ""]
    for tname in ["tumor", "loan"]:
        md += [f"## {tname}", "", "| method | accuracy | AUROC | worth about N examples | valid runs |",
               "|---|---|---|---|---|"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for kind, color in [("simple", "#1f77b4"), ("small", "#ff7f0e")]:
            curve = base[tname][kind]
            ys = [statistics.mean(r[0] for r in curve[f"train n={n}"]) for n in SIZES]
            ax.plot(SIZES, ys, "o-", color=color, label=f"training, {kind} net")
            md.append(f"| training, {kind} net | " + ", ".join(
                f"n={n}: {y:.1f}%" for n, y in zip(SIZES, ys)) + " | | | 20 each |")
        rnd = base[tname]["simple"]["random weights"]
        md.append(f"| random weights | {statistics.mean(r[0] for r in rnd):.1f}% | "
                  f"{statistics.mean(r[1] for r in rnd):.3f} | | 20 |")
        styles = {"weights_simple": ("#1f77b4", "--"), "weights_small": ("#ff7f0e", "--"),
                  "python_rule": ("#2ca02c", ":"), "hidden_names": ("#7f7f7f", "-.")}
        for cond in CONDITIONS:
            rs = [r for r in llm if r["task"] == tname and r["condition"] == cond]
            ok = [r for r in rs if r["valid"]]
            if not ok:
                md.append(f"| {label[cond]} | – | – | – | 0/{len(rs)} |"); continue
            a = [r["acc"] for r in ok]; u = [r["auc"] for r in ok]
            kind = "small" if cond == "weights_small" else "simple"
            worth = crossover(statistics.mean(a), base[tname][kind])
            md.append(f"| {label[cond]} | {statistics.mean(a):.1f}% ({min(a):.1f}–{max(a):.1f}) | "
                      f"{statistics.mean(u):.3f} | {worth} | {len(ok)}/{len(rs)} |")
            c, ls = styles[cond]
            ax.axhline(statistics.mean(a), color=c, ls=ls, lw=1.5, label=f"{label[cond]} (0 examples)")
        ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels(SIZES)
        ax.set_xlabel("labeled training examples"); ax.set_ylabel("test accuracy (%)")
        ax.set_title(f"{tname}: LLM with zero data vs training with n examples")
        ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(llm_dir / f"chart_{tname}.png", dpi=140); plt.close(fig)
        md.append("")
        for key, rows in comb.items():
            if not key.startswith(tname):
                continue
            md += [f"### {key}: start from LLM weights vs from scratch, then train", "",
                   "| examples | from LLM | from scratch |", "|---|---|---|"]
            for n, v in rows.items():
                md.append(f"| {n} | {statistics.mean(r[0] for r in v['from_llm']):.1f}% | "
                          f"{statistics.mean(r[0] for r in v['from_scratch']):.1f}% |")
            md.append("")
    (llm_dir / "report.md").write_text("\n".join(md))
    print("\n".join(md)); print(f"\nsaved {llm_dir / 'report.md'} and charts")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["baselines", "pilot", "llm", "combine", "report"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--run", default=None, help="LLM results folder (default: latest)")
    a = ap.parse_args()
    {"baselines": phase_baselines, "pilot": phase_pilot, "llm": phase_llm,
     "combine": phase_combine, "report": phase_report}[a.phase](a)
