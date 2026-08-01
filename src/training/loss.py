import numpy as np
import pandas as pd
import torch

from sklearn.utils.class_weight import compute_class_weight

from src.utils.config import LABEL_MAP


def create_weighted_loss(metadata_path, device):

    df = pd.read_csv(metadata_path)

    labels = df["dx"].map(LABEL_MAP)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    return criterion