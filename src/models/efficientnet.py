import timm
import torch.nn as nn


def build_model(num_classes=9):

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        num_classes
    )

    return model