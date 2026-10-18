import numpy as np
import torch

from sklearn.utils.class_weight import compute_class_weight

from src.utils.config import LABEL_MAP


def create_weighted_loss(
    train_labels,
    device,
    class_weight_power=0.5,
    label_smoothing=0.05,
):

    # ---------------------------------------------------
    # CONVERT LABELS TO NUMERIC VALUES
    # ---------------------------------------------------

    labels = train_labels.map(
        LABEL_MAP
    )

    if labels.isna().any():

        unknown_labels = (
            train_labels[
                labels.isna()
            ]
            .unique()
        )

        raise ValueError(
            f"Unmapped labels found: "
            f"{unknown_labels}"
        )

    labels = labels.astype(int)


    # ---------------------------------------------------
    # CLASS INDICES
    # ---------------------------------------------------

    classes = np.array(
        sorted(
            LABEL_MAP.values()
        )
    )


    # ---------------------------------------------------
    # ORIGINAL BALANCED WEIGHTS
    # ---------------------------------------------------

    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )


    # ---------------------------------------------------
    # TEMPER CLASS WEIGHTS
    # ---------------------------------------------------

    # Square-root weighting prevents very rare classes
    # from dominating the optimization.

    class_weights = (
        raw_weights
        ** class_weight_power
    )


    # ---------------------------------------------------
    # NORMALIZE WEIGHTS
    # ---------------------------------------------------

    class_weights = (
        class_weights
        / class_weights.mean()
    )


    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device,
    )


    # ---------------------------------------------------
    # PRINT WEIGHTS
    # ---------------------------------------------------

    print(
        "\nTempered Class Weights:"
    )

    for class_name, class_index in LABEL_MAP.items():

        print(
            f"{class_name:8s} : "
            f"{class_weights[class_index].item():.4f}"
        )


    # ---------------------------------------------------
    # LOSS
    # ---------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


    return criterion