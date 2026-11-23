"""
The two tasks for the "Can an LLM replace training data?" experiment.

Task A  tumor   : breast-cancer data built into scikit-learn (30 named measurements).
                  label 1 = malignant (cancer), 0 = benign.
Task B  loan    : an invented rule, written in plain English, used to generate fake
                  applicants. The LLM can't have seen this data anywhere.
                  label 1 = approved, 0 = rejected.

Each task gives: a training POOL (labeled examples that training may sample from),
a locked TEST set, feature names + descriptions, and the mean/std of each feature
computed on the pool WITHOUT labels (used to standardize inputs so 0 = average).
"""
import numpy as np

SEED = 0


def _split(X, y, test_frac, rng):
    idx = rng.permutation(len(y))
    n_test = int(round(test_frac * len(y)))
    te, po = idx[:n_test], idx[n_test:]
    return X[po], y[po], X[te], y[te]


def tumor_task():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    X = d.data.astype(np.float64)
    y = (d.target == 0).astype(np.int64)          # 1 = malignant
    rng = np.random.default_rng(SEED)
    Xp, yp, Xt, yt = _split(X, y, 0.3, rng)
    names = [str(n) for n in d.feature_names]
    context = (
        "Each case is a breast mass. A fine-needle aspirate was imaged and 10 properties of the cell "
        "nuclei were measured: radius, texture (variation in gray level), perimeter, area, smoothness "
        "(local variation in radius), compactness (perimeter^2/area - 1), concavity (severity of concave "
        "parts of the contour), concave points (number of concave portions), symmetry, and fractal "
        "dimension. For each property there are 3 features: 'mean' (average over the nuclei), 'error' "
        "(standard error), and 'worst' (mean of the three largest values)."
    )
    question = "Predict whether the mass is MALIGNANT (cancer). Output = probability of malignant."
    return dict(name="tumor", Xp=Xp, yp=yp, Xt=Xt, yt=yt, names=names, context=context,
                question=question, rule=None)


LOAN_FEATURES = [
    ("income_to_loan_ratio", "yearly income divided by the loan amount"),
    ("debt_to_income", "monthly debt payments divided by monthly income (0.35 = 35%)"),
    ("years_employed", "years at the current job"),
    ("missed_payments_2y", "number of missed loan/credit-card payments in the last 2 years"),
    ("credit_history_years", "years since the applicant's first credit account"),
    ("age", "age in years"),
    ("num_dependents", "number of children or other dependents"),
    ("savings_k", "savings in thousands of dollars"),
]

LOAN_RULE = (
    "The bank approves a loan only if ALL of these hold: income_to_loan_ratio is above 2.0, "
    "debt_to_income is below 0.40, years_employed is at least 2, and missed_payments_2y is below 2. "
    "Otherwise it rejects. The other features play no role in the decision. About 5% of decisions "
    "are random clerical errors."
)


def loan_applicants(n, seed, shifted=False):
    """Generate applicants and label them with the rule (+5% random errors).
    shifted=True draws from a different population (richer, older, more debt, longer jobs,
    more missed payments); the approval rule itself is unchanged."""
    rng = np.random.default_rng(seed)
    if not shifted:
        ratio = rng.lognormal(mean=np.log(2.6), sigma=0.45, size=n)
        dti = np.clip(rng.normal(0.32, 0.12, n), 0.0, 0.9)
        years = np.clip(rng.exponential(4.0, n), 0, 40)
        missed = rng.poisson(0.6, n).astype(np.float64)
        history = np.clip(rng.normal(10, 6, n), 0, 50)
        age = np.clip(rng.normal(40, 12, n), 18, 85)
        deps = rng.poisson(1.2, n).astype(np.float64)
        savings = np.clip(rng.lognormal(np.log(15), 1.0, n), 0, 500)
    else:
        ratio = rng.lognormal(mean=np.log(3.2), sigma=0.6, size=n)
        dti = np.clip(rng.normal(0.38, 0.15, n), 0.0, 0.9)
        years = np.clip(rng.exponential(7.0, n), 0, 40)
        missed = rng.poisson(1.1, n).astype(np.float64)
        history = np.clip(rng.normal(18, 8, n), 0, 50)
        age = np.clip(rng.normal(52, 10, n), 18, 85)
        deps = rng.poisson(2.0, n).astype(np.float64)
        savings = np.clip(rng.lognormal(np.log(40), 1.0, n), 0, 500)
    X = np.stack([ratio, dti, years, missed, history, age, deps, savings], 1)
    y = ((ratio > 2.0) & (dti < 0.40) & (years >= 2) & (missed < 2)).astype(np.int64)
    flip = rng.random(n) < 0.05
    return X, np.where(flip, 1 - y, y)


def loan_task(n=2000):
    # original generator, kept byte-for-byte so earlier results stay valid
    rng = np.random.default_rng(SEED + 1)
    ratio = rng.lognormal(mean=np.log(2.6), sigma=0.45, size=n)
    dti = np.clip(rng.normal(0.32, 0.12, n), 0.0, 0.9)
    years = np.clip(rng.exponential(4.0, n), 0, 40)
    missed = rng.poisson(0.6, n).astype(np.float64)
    history = np.clip(rng.normal(10, 6, n), 0, 50)
    age = np.clip(rng.normal(40, 12, n), 18, 85)
    deps = rng.poisson(1.2, n).astype(np.float64)
    savings = np.clip(rng.lognormal(np.log(15), 1.0, n), 0, 500)
    X = np.stack([ratio, dti, years, missed, history, age, deps, savings], 1)
    y = ((ratio > 2.0) & (dti < 0.40) & (years >= 2) & (missed < 2)).astype(np.int64)
    flip = rng.random(n) < 0.05
    y = np.where(flip, 1 - y, y)
    Xp, yp, Xt, yt = _split(X, y, 0.3, rng)
    names = [f for f, _ in LOAN_FEATURES]
    context = "Each case is a loan applicant. Features:\n" + "\n".join(
        f"- {f}: {d}" for f, d in LOAN_FEATURES)
    question = "Predict whether the loan is APPROVED. Output = probability of approval."
    return dict(name="loan", Xp=Xp, yp=yp, Xt=Xt, yt=yt, names=names, context=context,
                question=question, rule=LOAN_RULE)


def standardize(task):
    mu = task["Xp"].mean(0); sd = task["Xp"].std(0) + 1e-12   # label-free statistics
    task["mu"], task["sd"] = mu, sd
    task["Zp"] = (task["Xp"] - mu) / sd
    task["Zt"] = (task["Xt"] - mu) / sd
    return task


def get_tasks():
    return [standardize(tumor_task()), standardize(loan_task())]


if __name__ == "__main__":
    for t in get_tasks():
        print(t["name"], "pool", t["Xp"].shape, "positives %.2f" % t["yp"].mean(),
              "| test", t["Xt"].shape, "positives %.2f" % t["yt"].mean())
