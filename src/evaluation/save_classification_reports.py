import json
from pathlib import Path
import timm
import torch
from sklearn.metrics import classification_report
from src.preprocessing.create_dataloader import create_dataloaders
from src.training.train_config import DEVICE
from src.utils.config import INDEX_TO_LABEL

OUT = Path("outputs/reports")
OUT.mkdir(parents=True, exist_ok=True)

_, _, test_loader = create_dataloaders(batch_size=4)
names = [INDEX_TO_LABEL[i] for i in range(9)]

models = [
    ("EfficientNet-B0", "efficientnet_b0",
     "outputs/models/best_model_b0_final.pth", "b0"),
    ("EfficientNet-B4", "efficientnet_b4",
     "outputs/models/best_model_efficientnet_b4.pth", "b4"),
]

for name, architecture, checkpoint_path, tag in models:
    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )
    model = timm.create_model(
        architecture, pretrained=False, num_classes=9
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(DEVICE))
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())

    report = classification_report(
        y_true, y_pred,
        labels=list(range(9)),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    output_file = OUT / f"classification_report_{tag}.json"
    output_file.write_text(json.dumps({
        "model": name,
        "checkpoint": checkpoint_path,
        "test_samples": len(y_true),
        "classification_report": report,
    }, indent=2))

    print(f"{name}: saved {output_file}")

print("Classification reports generated.")
