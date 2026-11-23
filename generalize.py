"""
Generalization check on large and shifted test sets (no LLM calls).

Re-scores the networks the LLM already wrote, plus normally trained networks:
  loan  : 100,000 NEW applicants (same population) and 100,000 from a SHIFTED population
          (richer, older, more debt); the approval rule is unchanged.
  tumor : all 569 real cases (the LLM used none of them, so every case is unseen).
Inputs are standardized with the ORIGINAL statistics the LLM was given.

Usage: .venv/bin/python experiments/knowledge_vs_data/generalize.py [llm results folder]
"""
import json, statistics, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R                                    # noqa: E402
from tasks import get_tasks, loan_applicants       # noqa: E402

llm_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else R.latest("llm")
tasks = {t["name"]: t for t in get_tasks()}
results = json.loads((llm_dir / "llm_results.json").read_text())

sets = {}
t = tasks["loan"]
for label, seed, shifted in [("loan: 100k new applicants", 12345, False),
                             ("loan: 100k SHIFTED population", 54321, True)]:
    X, y = loan_applicants(100_000, seed, shifted)
    sets[label] = ("loan", X, y)
tt = tasks["tumor"]
sets["tumor: all 569 real cases"] = ("tumor", np.vstack([tt["Xp"], tt["Xt"]]), np.concatenate([tt["yp"], tt["yt"]]))

lines = [f"# Generalization check ({llm_dir.name})", ""]
for label, (tname, X, y) in sets.items():
    t = tasks[tname]; Z = (X - t["mu"]) / t["sd"]
    lines += [f"## {label} (positives {100 * y.mean():.0f}%)", "",
              "| method | accuracy | AUROC | runs |", "|---|---|---|---|"]
    lines.append(f"| always guess the common answer | {100 * max(y.mean(), 1 - y.mean()):.1f}% | 0.500 | – |")
    # normally trained networks
    for kind in ["simple", "small"]:
        for n in [250, "all"]:
            accs, aucs = [], []
            reps = 1 if n == "all" else 10
            rng = np.random.default_rng(7); gen = torch.Generator().manual_seed(7)
            for _ in range(reps):
                idx = np.arange(len(t["yp"])) if n == "all" else R.sample_idx(t["yp"], n, rng)
                p = R.train(R.init_params(kind, Z.shape[1], gen), t["Zp"][idx], t["yp"][idx])
                a, u = R.evaluate(p, Z, y); accs.append(a); aucs.append(u)
            nn = f"all {len(t['yp'])}" if n == "all" else n
            if tname == "tumor" and n == "all":
                nn += " (includes some of these cases)"
            lines.append(f"| trained {kind} net, {nn} examples | {statistics.mean(accs):.1f}% | "
                         f"{statistics.mean(aucs):.3f} | {reps} |")
    # LLM-written networks and rules
    for cond, lab in [("weights_simple", "LLM weights, simple net"), ("weights_small", "LLM weights, small net"),
                      ("python_rule", "LLM Python rule")]:
        accs, aucs = [], []
        for r in results:
            if r["task"] != tname or r["condition"] != cond or not r["valid"]:
                continue
            run_dir = llm_dir / "runs" / f"{tname}_{cond}_r{r['repeat']}"
            if cond == "python_rule":
                sc = R.score_python_rule((run_dir / "reply.txt").read_text(),
                                         {"names": t["names"], "Xt": X, "yt": y})
            else:
                sc = R.evaluate(torch.load(run_dir / "params.pt"), Z, y)
            if sc:
                accs.append(sc[0]); aucs.append(sc[1])
        if accs:
            lines.append(f"| **{lab} (0 examples)** | {statistics.mean(accs):.1f}% "
                         f"({min(accs):.1f}–{max(accs):.1f}) | {statistics.mean(aucs):.3f} | {len(accs)} |")
    lines.append("")

out = "\n".join(lines)
(llm_dir / "generalization.md").write_text(out)
print(out)
