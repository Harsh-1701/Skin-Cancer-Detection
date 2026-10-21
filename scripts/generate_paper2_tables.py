import json
from pathlib import Path

import pandas as pd

OUT = Path("outputs/paper2")
METRICS = OUT / "error_detection_metrics_model_specific_preprocessing.json"

METHODS = ["confidence_only", "disagreement_only", "combined"]
LABELS = {
    "confidence_only": "confidence_only",
    "disagreement_only": "disagreement_only",
    "combined": "combined",
}

with METRICS.open("r", encoding="utf-8") as f:
    data = json.load(f)

test_metrics = data["test_metrics"]
comparison_rows = []
result_rows = []

for model in ("B0", "B4"):
    if model not in test_metrics:
        raise KeyError(f"Missing model: {model}")

    confidence = test_metrics[model]["confidence_only"]

    for method in METHODS:
        if method not in test_metrics[model]:
            raise KeyError(f"Missing method: {model}/{method}")

        m = test_metrics[model][method]

        comparison_rows.append({
            "Model": model,
            "Method": LABELS[method],
            "AUROC": m["AUROC"],
            "AUPRC": m["AUPRC"],
            "Top-20% error precision": m["flagged_error_precision"],
            "Top-20% error recall": m["flagged_error_recall"],
            "Top-20% error F1": m["flagged_error_f1"],
            "Change vs confidence-only: AUROC":
                m["AUROC"] - confidence["AUROC"]
                if method != "confidence_only" else None,
            "Change vs confidence-only: AUPRC":
                m["AUPRC"] - confidence["AUPRC"]
                if method != "confidence_only" else None,
            "Change vs confidence-only: Top-20% error precision":
                m["flagged_error_precision"] - confidence["flagged_error_precision"]
                if method != "confidence_only" else None,
            "Change vs confidence-only: Top-20% error recall":
                m["flagged_error_recall"] - confidence["flagged_error_recall"]
                if method != "confidence_only" else None,
            "Change vs confidence-only: Top-20% error F1":
                m["flagged_error_f1"] - confidence["flagged_error_f1"]
                if method != "confidence_only" else None,
        })

        result_rows.append({
            "Model": model,
            "Method": LABELS[method],
            "AUROC": m["AUROC"],
            "AUPRC": m["AUPRC"],
            "Top-20% error precision": m["flagged_error_precision"],
            "Top-20% error recall": m["flagged_error_recall"],
            "Top-20% error F1": m["flagged_error_f1"],
            "Flagged error rate": m["error_rate_among_flagged"],
            "Unflagged error rate": m["error_rate_among_unflagged"],
        })

comparison_path = OUT / "paper2_final_comparison.csv"
results_path = OUT / "paper2_test_results_table.csv"

pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False, float_format="%.10f")
pd.DataFrame(result_rows).to_csv(results_path, index=False, float_format="%.10f")

# Read back and verify the generated tables against the source metrics.
for path in (comparison_path, results_path):
    df = pd.read_csv(path)
    assert len(df) == 6, f"{path.name}: expected 6 rows, got {len(df)}"
    assert set(df["Model"]) == {"B0", "B4"}
    assert set(df["Method"]) == set(METHODS)
    print(f"Verified {path}: {len(df)} rows")

print("\nCurrent corrected test results:")
for model in ("B0", "B4"):
    for method in METHODS:
        m = test_metrics[model][method]
        print(
            f"{model:2} {method:18} "
            f"AUROC={m['AUROC']:.4f} "
            f"AUPRC={m['AUPRC']:.4f} "
            f"P@20={m['flagged_error_precision']:.4f} "
            f"R@20={m['flagged_error_recall']:.4f} "
            f"F1@20={m['flagged_error_f1']:.4f}"
        )
