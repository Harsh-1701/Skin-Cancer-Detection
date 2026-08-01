import cv2
import torch

from torchvision import transforms

from src.models.efficientnet import build_model

from src.utils.config import (
    CLASS_INFO,
    INDEX_TO_LABEL,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_model(model_path):

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    model = build_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model


def predict(image_path, model):

    image = cv2.imread(image_path)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    disease = INDEX_TO_LABEL[
        prediction.item()
    ]

    print("\nPrediction")
    print("-" * 40)

    print(
        f"Disease : {disease}"
    )

    print(
        f"Type : {CLASS_INFO[disease]}"
    )

    print(
        f"Confidence : {confidence.item()*100:.2f}%"
    )


def main():

    model = load_model(
        "outputs/models/best_model.pth"
    )

    image_path = input(
        "Enter image path: "
    )

    predict(
        image_path,
        model
    )


if __name__ == "__main__":
    main()