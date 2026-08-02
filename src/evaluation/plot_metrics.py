import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import OUTPUT_DIR

OUTPUT_DIR.mkdir(exist_ok=True)

labels = [
    "AKIEC",
    "BCC",
    "BKL",
    "DF",
    "MEL",
    "NV",
    "VASC",
]

cm = np.load(
    "outputs/confusion_matrix.npy"
)

plt.figure(figsize=(8,7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()