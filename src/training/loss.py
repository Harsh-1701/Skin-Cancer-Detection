import numpy as np
import pandas as pd
import torch

from sklearn.utils.class_weight import compute_class_weight

from src.utils.config import LABEL_MAP


def create_weighted_loss(dataset_path, device):

    # ---------------------------------------------------
    # LOAD COMBINED DATASET
    # ---------------------------------------------------

    df = pd.read_csv(dataset_path)

    # Convert class names to numeric labels
    labels = df["dx"].map(LABEL_MAP)

    # Make sure there are no unmapped labels
    if labels.isna().any():

        unknown_labels = df.loc[
            labels.isna(),
            "dx"
        ].unique()

        raise ValueError(
            f"Unmapped labels found: {unknown_labels}"
        )

    labels = labels.astype(int)

    # ---------------------------------------------------
    # CALCULATE CLASS WEIGHTS
    # ---------------------------------------------------

    classes = np.array(
        sorted(LABEL_MAP.values())
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )

    print("\nClass Weights:")

    for class_name, class_index in LABEL_MAP.items():

        print(
            f"{class_name:8s} : "
            f"{class_weights[class_index].item():.4f}"
        )

    # ---------------------------------------------------
    # WEIGHTED CROSS-ENTROPY LOSS
    # ---------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    return criterion