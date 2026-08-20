import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.models.efficientnet import build_model

from src.preprocessing.create_dataloader import (
    create_dataloaders
)

from src.training.train_config import DEVICE

from src.utils.config import (
    INDEX_TO_LABEL,
)


# -------------------------------------------------------
# OUTPUT DIRECTORY
# -------------------------------------------------------

PLOT_DIR = Path(
    "outputs/plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# -------------------------------------------------------
# EVALUATION
# -------------------------------------------------------

def evaluate():

    print("=" * 60)
    print("Skin Cancer Detection System - V2")
    print("B0 Test Set Evaluation")
    print("=" * 60)

    print(
        "\nDevice:",
        DEVICE
    )

    # ---------------------------------------------------
    # CREATE DATA LOADERS
    # ---------------------------------------------------

    _, _, test_loader = create_dataloaders(
        batch_size=32
    )

    # ---------------------------------------------------
    # LOAD BEST MODEL
    # ---------------------------------------------------

    checkpoint = torch.load(
        "outputs/models/best_model.pth",
        map_location=DEVICE,
    )

    model = build_model(
        num_classes=9
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print(
        "\nBest model loaded."
    )

    print(
        "Best validation accuracy:",
        f"{checkpoint['validation_accuracy']:.2f}%"
    )

    # ---------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------

    y_true = []

    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            predictions = (
                outputs
                .argmax(1)
                .cpu()
                .numpy()
            )

            y_pred.extend(
                predictions
            )

            y_true.extend(
                labels.numpy()
            )

    # ---------------------------------------------------
    # CLASS NAMES
    # ---------------------------------------------------

    num_classes = 9

    target_names = [
        INDEX_TO_LABEL[i]
        for i in range(num_classes)
    ]

    # ---------------------------------------------------
    # OVERALL METRICS
    # ---------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # ---------------------------------------------------
    # PRINT OVERALL RESULTS
    # ---------------------------------------------------

    print("\n" + "=" * 60)

    print("TEST SET RESULTS")

    print("=" * 60)

    print(
        f"Accuracy          : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Weighted Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Weighted Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{f1:.4f}"
    )

    print(
        f"Macro Precision    : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall       : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{macro_f1:.4f}"
    )

    print("=" * 60)

    # ---------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------

    print(
        "\nCLASSIFICATION REPORT"
    )

    print("=" * 60)

    report = classification_report(

        y_true,

        y_pred,

        labels=list(
            range(num_classes)
        ),

        target_names=target_names,

        digits=4,

        zero_division=0,
    )

    print(report)

    # ---------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=list(
            range(num_classes)
        ),
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    # ---------------------------------------------------
    # SAVE CONFUSION MATRIX
    # ---------------------------------------------------

    plt.figure(
        figsize=(11, 9)
    )

    plt.imshow(
        cm,
        cmap="Blues",
    )

    plt.title(
        "EfficientNet-B0 - V2 Test Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "Actual Class"
    )

    plt.xticks(
        range(num_classes),
        target_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(num_classes),
        target_names,
    )

    # ---------------------------------------------------
    # WRITE VALUES
    # ---------------------------------------------------

    threshold = cm.max() / 2

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(

                j,

                i,

                cm[i, j],

                ha="center",

                va="center",

                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
            )

    plt.colorbar()

    plt.tight_layout()

    confusion_path = (
        PLOT_DIR
        / "confusion_matrix_b0_test.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nConfusion matrix saved:"
    )

    print(
        confusion_path
    )

    print(
        "\nEvaluation completed."
    )


# -------------------------------------------------------
# RUN
# -------------------------------------------------------

if __name__ == "__main__":

    evaluate()