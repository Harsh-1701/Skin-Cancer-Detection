
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("outputs/paper2")
PRED = ROOT / "test_predictions_model_specific_preprocessing.csv"
OUT = ROOT / "paired_method_comparisons.json"
N_BOOT = 2000
SEED = 123
TOP_FRACTION = 0.20


def metric_values(y, risk):
    if np.unique(y).size < 2:
        return None

    n_flag = max(1, int(np.ceil(TOP_FRACTION * len(y))))
    order = np.argsort(-risk, kind="stable")
    flagged = np.zeros(len(y), dtype=bool)
    flagged[order[:n_flag]] = True

    precision = float(y[flagged].mean())
    recall = float(y[flagged].sum() / y.sum()) if y.sum() else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "AUROC": float(roc_auc_score(y, risk)),
        "AUPRC": float(average_precision_score(y, risk)),
        "top20_precision": precision,
        "top20_recall": recall,
        "top20_F1": float(f1),
    }


def main():
    df = pd.read_csv(PRED)
    rng = np.random.default_rng(SEED)
    n = len(df)

    disagree = df["models_disagree"].astype(float).to_numpy()
    configs = {
        "B0": {
            "y": (~df["b0_correct"].astype(bool)).astype(int).to_numpy(),
            "confidence": 1 - df["b0_confidence"].to_numpy(),
            "alpha": 0.92,
        },
        "B4": {
            "y": (~df["b4_correct"].astype(bool)).astype(int).to_numpy(),
            "confidence": 1 - df["b4_confidence"].to_numpy(),
            "alpha": 0.55,
        },
    }

    output = {
        "metadata": {
            "n_test": n,
            "bootstrap_replicates": N_BOOT,
            "seed": SEED,
            "method": "paired percentile bootstrap over test rows",
            "confidence_level": 0.95,
            "positive_difference_means": "first method performs better",
        },
        "comparisons": {},
    }

    for model, cfg in configs.items():
        y = cfg["y"]
        conf = cfg["confidence"]
        combined = cfg["alpha"] * conf + (1 - cfg["alpha"]) * disagree

        risks = {
            "confidence_only": conf,
            "disagreement_only": disagree,
            "combined": combined,
        }

        # Compare combined vs confidence, and disagreement vs confidence.
        pairs = [
            ("combined", "confidence_only"),
            ("disagreement_only", "confidence_only"),
        ]
        output["comparisons"][model] = {}

        for first, second in pairs:
            point_a = metric_values(y, risks[first])
            point_b = metric_values(y, risks[second])
            point_diff = {
                key: point_a[key] - point_b[key] for key in point_a
            }

            boot_diffs = {key: [] for key in point_diff}
            accepted = 0
            attempts = 0

            while accepted < N_BOOT and attempts < N_BOOT * 3:
                attempts += 1
                idx = rng.integers(0, n, size=n)
                ma = metric_values(y[idx], risks[first][idx])
                mb = metric_values(y[idx], risks[second][idx])

                if ma is None or mb is None:
                    continue

                for key in boot_diffs:
                    boot_diffs[key].append(ma[key] - mb[key])
                accepted += 1

            if accepted < N_BOOT:
                raise RuntimeError(
                    f"Insufficient valid replicates for {model}: {first} vs {second}"
                )

            summary = {}
            for key, estimate in point_diff.items():
                low, high = np.percentile(boot_diffs[key], [2.5, 97.5])
                summary[key] = {
                    "difference": estimate,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "bootstrap_fraction_positive": float(
                        np.mean(np.array(boot_diffs[key]) > 0)
                    ),
                }

            output["comparisons"][model][f"{first}_minus_{second}"] = summary
            print(f"Completed {model}: {first} minus {second}")

    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved paired comparisons: {OUT}")


if __name__ == "__main__":
    main()
