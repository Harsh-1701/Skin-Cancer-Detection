
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("outputs/paper2")
PREDICTIONS = ROOT / "test_predictions_model_specific_preprocessing.csv"
OUTPUT = ROOT / "bootstrap_confidence_intervals.json"

N_BOOTSTRAPS = 2000
SEED = 42
FLAG_FRACTION = 0.20

def metrics_for_sample(y, risk):
    """Calculate error-detection metrics; larger risk means more likely error."""
    n = len(y)
    if len(np.unique(y)) < 2:
        return None

    k = max(1, int(np.ceil(FLAG_FRACTION * n)))
    flagged = np.argsort(-risk, kind="stable")[:k]
    flags = np.zeros(n, dtype=bool)
    flags[flagged] = True

    tp = int(np.sum(y[flags] == 1))
    precision = tp / k
    recall = tp / int(y.sum()) if y.sum() else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "AUROC": float(roc_auc_score(y, risk)),
        "AUPRC": float(average_precision_score(y, risk)),
        "top20_precision": float(precision),
        "top20_recall": float(recall),
        "top20_F1": float(f1),
    }

def main():
    df = pd.read_csv(PREDICTIONS)
    required = {
        "b0_correct", "b4_correct",
        "b0_confidence", "b4_confidence",
        "models_disagree",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    # Error = 1; risk scores must increase as predicted error risk increases.
    targets = {
        "B0": (~df["b0_correct"].astype(bool)).astype(int).to_numpy(),
        "B4": (~df["b4_correct"].astype(bool)).astype(int).to_numpy(),
    }
    disagreement = df["models_disagree"].astype(float).to_numpy()

    # Match the evaluation's combined-score definition and validation weights.
    weights = {"B0": 0.92, "B4": 0.55}
    risks = {
        "B0": {
            "confidence_only": 1 - df["b0_confidence"].to_numpy(),
            "disagreement_only": disagreement,
            "combined": (
                weights["B0"] * (1 - df["b0_confidence"].to_numpy())
                + (1 - weights["B0"]) * disagreement
            ),
        },
        "B4": {
            "confidence_only": 1 - df["b4_confidence"].to_numpy(),
            "disagreement_only": disagreement,
            "combined": (
                weights["B4"] * (1 - df["b4_confidence"].to_numpy())
                + (1 - weights["B4"]) * disagreement
            ),
        },
    }

    rng = np.random.default_rng(SEED)
    n = len(df)
    results = {
        "metadata": {
            "n_test": n,
            "n_bootstraps_requested": N_BOOTSTRAPS,
            "seed": SEED,
            "confidence_level": 0.95,
            "method": "paired percentile bootstrap over test rows",
            "top_fraction": FLAG_FRACTION,
        },
        "metrics": {},
    }

    for model in ("B0", "B4"):
        y = targets[model]
        results["metrics"][model] = {}

        for method, risk in risks[model].items():
            point = metrics_for_sample(y, risk)
            if point is None:
                raise ValueError(f"Test target for {model} has only one class.")

            samples = {key: [] for key in point}
            accepted = 0
            attempts = 0
            max_attempts = N_BOOTSTRAPS * 3

            while accepted < N_BOOTSTRAPS and attempts < max_attempts:
                attempts += 1
                idx = rng.integers(0, n, size=n)
                sampled = metrics_for_sample(y[idx], risk[idx])
                if sampled is None:
                    continue
                for key, value in sampled.items():
                    samples[key].append(value)
                accepted += 1

            if accepted < N_BOOTSTRAPS:
                raise RuntimeError(
                    f"Only {accepted} valid bootstrap samples for {model}/{method}"
                )

            intervals = {}
            for key, values in samples.items():
                low, high = np.percentile(values, [2.5, 97.5])
                intervals[key] = {
                    "estimate": point[key],
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                }

            results["metrics"][model][method] = {
                "n_errors": int(y.sum()),
                "metrics": intervals,
            }
            print(f"Completed {model} / {method}")

    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved confidence intervals to: {OUTPUT}")

if __name__ == "__main__":
    main()
