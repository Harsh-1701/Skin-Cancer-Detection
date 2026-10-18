import timm
import torch.nn as nn


def build_model(num_classes=9):

    model = timm.create_model(
        "efficientnet_b4",
        pretrained=True,
        num_classes=num_classes,
    )

    return model