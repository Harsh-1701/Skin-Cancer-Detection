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
from src.preprocessing.create_dataloader import create_dataloaders
from src.training.train_config import DEVICE
from src.utils.config import INDEX_TO_LABEL


# -------------------------------------------------------
# OUTPUT DIRECTORY
# -------------------------------------------------------

Path("outputs/plots").mkdir(
    parents=True,
    exist_ok=True,
)


# -------------------------------------------------------
# B4 EVALUATION
# -------------------------------------------------------

def evaluate():

    print("=" * 60)
    print("Skin Cancer Detection System - V2")
    print("EfficientNet-B4 Test Evaluation")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    _, _, test_loader = create_dataloaders(
        batch_size=4
    )

    # ---------------------------------------------------
    # LOAD B4 BEST CHECKPOINT
    # ---------------------------------------------------

    checkpoint_path = (
        "outputs/models/best_model_b0_improved.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    # ---------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------

    model = build_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    print("\nB4 best model loaded.")

    if "best_val_accuracy" in checkpoint:
        print(
            f"Best validation accuracy: "
            f"{checkpoint['best_val_accuracy']:.2f}%"
        )

    # ---------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = (
                outputs.argmax(1)
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

    target_names = [
        INDEX_TO_LABEL[i]
        for i in range(9)
    ]

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    weighted_precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
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

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # ---------------------------------------------------
    # RESULTS
    # ---------------------------------------------------

    print("\n" + "=" * 60)
    print("B4 TEST SET RESULTS")
    print("=" * 60)

    print(
        f"Accuracy           : {accuracy * 100:.2f}%"
    )

    print(
        f"Weighted Precision : {weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall    : {weighted_recall:.4f}"
    )

    print(
        f"Weighted F1        : {weighted_f1:.4f}"
    )

    print(
        f"Macro Precision    : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall       : {macro_recall:.4f}"
    )

    print(
        f"Macro F1           : {macro_f1:.4f}"
    )

    print("=" * 60)

    # ---------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------

    print("\nCLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    # ---------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    print("Confusion Matrix:")
    print(cm)

    # ---------------------------------------------------
    # PLOT
    # ---------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        cm,
        cmap="Blues",
    )

    plt.title(
        "EfficientNet-B4 Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        range(9),
        target_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(9),
        target_names,
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
            )

    plt.colorbar()

    plt.tight_layout()

    output_path = (
        "outputs/plots/"
        "confusion_matrix_b4_test.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nConfusion matrix saved:"
    )

    print(output_path)

    print("\nB4 evaluation completed.")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

if __name__ == "__main__":

    evaluate()