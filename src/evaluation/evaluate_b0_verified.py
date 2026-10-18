from pathlib import Path
import timm
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing.create_dataloader import create_dataloaders
from src.training.train_config import DEVICE
from src.utils.config import INDEX_TO_LABEL

def evaluate():
    checkpoint_path = Path("outputs/models/best_model_b0_final.pth")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=9)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    _, _, test_loader = create_dataloaders(batch_size=32)
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())

    names = [INDEX_TO_LABEL[i] for i in range(9)]
    print("\nB0 TEST RESULTS")
    print("Checkpoint:", checkpoint_path)
    print(f"Validation accuracy: {checkpoint['validation_accuracy']:.2f}%")
    print(f"Test accuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Weighted precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Weighted recall: {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Macro precision: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro recall: {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print("\nClassification report:\n", classification_report(y_true, y_pred, labels=list(range(9)), target_names=names, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(9)))
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(title="EfficientNet-B0 Test Confusion Matrix", xlabel="Predicted Class", ylabel="Actual Class")
    ax.set_xticks(range(9), names, rotation=45, ha="right")
    ax.set_yticks(range(9), names)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out = Path("outputs/plots/confusion_matrix_b0_verified.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Confusion matrix saved:", out)

if __name__ == "__main__":
    evaluate()
