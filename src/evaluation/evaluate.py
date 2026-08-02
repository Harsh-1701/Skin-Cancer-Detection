import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs/plots").mkdir(
    parents=True,
    exist_ok=True,
)

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


def evaluate():

    _, val_loader, _ = create_dataloaders(batch_size=32)

    checkpoint = torch.load(
        "outputs/models/best_model.pth",
        map_location=DEVICE,
    )

    model = build_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(1).cpu().numpy()

            y_pred.extend(predictions)

            y_true.extend(labels.numpy())

    print("=" * 60)

    print(
        f"Accuracy : {accuracy_score(y_true,y_pred)*100:.2f}%"
    )

    print(
        f"Precision : {precision_score(y_true,y_pred,average='weighted'):.4f}"
    )

    print(
        f"Recall : {recall_score(y_true,y_pred,average='weighted'):.4f}"
    )

    print(
        f"F1 Score : {f1_score(y_true,y_pred,average='weighted'):.4f}"
    )

    print("=" * 60)

    target_names = [
        INDEX_TO_LABEL[i]
        for i in range(7)
    ]

    print(

        classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
        )

    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    plt.figure(figsize=(8,6))

    plt.imshow(
        cm,
        cmap="Blues",
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.xticks(
        range(7),
        target_names,
    )

    plt.yticks(
        range(7),
        target_names,
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                cm[i,j],
                ha="center",
                va="center",
                color="black",
            )

    plt.colorbar()

    plt.tight_layout()

    plt.savefig(
        "outputs/plots/confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    print("Confusion matrix saved.")

    print("Confusion matrix saved.")


if __name__ == "__main__":

    evaluate()