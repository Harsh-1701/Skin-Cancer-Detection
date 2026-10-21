
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)

OUT = Path("outputs/paper2")
PRED = OUT / "test_predictions_model_specific_preprocessing.csv"
METRICS = OUT / "error_detection_metrics_model_specific_preprocessing.json"



def get_alpha(metrics, model):
    matches = []

    def search(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                key_lower = key.lower()

                if model.lower() in new_path.lower() and "alpha" in key_lower:
                    try:
                        matches.append(float(value))
                    except (TypeError, ValueError):
                        pass

                search(value, new_path)

        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                search(value, f"{path}[{i}]")

    search(metrics)

    valid = [x for x in matches if 0 <= x <= 1]
    if valid:
        return valid[0]

    raise KeyError(
        f"No saved alpha found for {model}. "
        "Use the validation alpha from the evaluator output."
    )



def get_disagreement(df):
    for col in ("models_disagree", "model_disagreement", "disagreement"):
        if col in df.columns:
            return df[col].astype(str).str.lower().map(
                {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
            ).to_numpy()

    for col in ("b0_prediction", "b4_prediction", "b0_pred", "b4_pred"):
        pass

    raise KeyError(
        "No disagreement column found. Available columns: "
        + ", ".join(df.columns)
    )


def curves_for_model(df, metrics, model):
    prefix = model.lower()
    correct_col = f"{prefix}_correct"
    confidence_col = f"{prefix}_confidence"

    y_correct = df[correct_col].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if y_correct.isna().any():
        raise ValueError(f"Could not parse {correct_col}")

    y_error = (~y_correct.to_numpy()).astype(int)
    confidence = df[confidence_col].astype(float).to_numpy()
    disagreement = get_disagreement(df)

    if not np.isfinite(confidence).all() or not np.isfinite(disagreement).all():
        raise ValueError("Predictions contain missing or non-finite scores.")

    alpha = get_alpha(metrics, model)
    if not 0 <= alpha <= 1:
        raise ValueError(f"Invalid alpha for {model}: {alpha}")

    scores = {
        "Confidence only": 1.0 - confidence,
        "Disagreement only": disagreement,
        "Combined": alpha * (1.0 - confidence) + (1.0 - alpha) * disagreement,
    }
    return y_error, scores


def save_roc(y, scores, model):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y, score):.3f})")
    ax.plot([0, 1], [0, 1], "--", label="Chance")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"{model}: Error-detection ROC",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"figure_{model.lower()}_roc_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_pr(y, scores, model):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, score in scores.items():
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        ax.plot(recall, precision, label=f"{name} (AUPRC {ap:.3f})")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"{model}: Error-detection precision–recall",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / f"figure_{model.lower()}_pr_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)



def save_risk_coverage(y, scores, model):
    fig, ax = plt.subplots(figsize=(7, 6))
    n = len(y)

    for name, score in scores.items():
        # Retain the lowest-risk predictions first.
        order = np.argsort(score, kind="stable")
        sorted_errors = y[order]

        coverage = np.arange(1, n + 1) / n
        risk = np.cumsum(sorted_errors) / np.arange(1, n + 1)
        aurc = auc(coverage, risk)

        ax.plot(coverage, risk, label=f"{name} (AURC {aurc:.3f})")

    ax.set(
        xlabel="Coverage (fraction of predictions retained)",
        ylabel="Error rate among retained predictions",
        title=f"{model}: Risk–coverage",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(
        OUT / f"figure_{model.lower()}_risk_coverage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)



def save_top20(metrics):
    methods = ["confidence_only", "disagreement_only", "combined"]
    labels = ["Confidence only", "Disagreement only", "Combined"]
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, model in enumerate(("B0", "B4")):
        values = [
            metrics["test_metrics"][model][method]["flagged_error_precision"]
            for method in methods
        ]
        ax.bar(x + (i - 0.5) * width, values, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Error precision among top 20% flagged")
    ax.set_ylim(0, 1)
    ax.set_title("Error detection at a fixed review budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figure_top20_error_precision.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if not PRED.is_file() or not METRICS.is_file():
        raise FileNotFoundError("Required prediction or metrics file is missing.")

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED)
    with METRICS.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    required = {
        "sample_id", "b0_confidence", "b0_correct",
        "b4_confidence", "b4_correct",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    for model in ("B0", "B4"):
        y, scores = curves_for_model(df, metrics, model)
        save_roc(y, scores, model)
        save_pr(y, scores, model)
        save_risk_coverage(y, scores, model)

    save_top20(metrics)
    print("Generated comparison figures in:", OUT.resolve())


if __name__ == "__main__":
    main()
