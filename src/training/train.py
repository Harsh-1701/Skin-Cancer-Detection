import torch

from torch.optim import AdamW

from src.models.efficientnet import build_model
from src.preprocessing.create_dataloader import create_dataloaders
from src.training.loss import create_weighted_loss
from src.training.train_config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from src.training.trainer import Trainer
from src.utils.config import (
    METADATA_PATH,
    MODEL_DIR,
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = build_model()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = create_weighted_loss(
        METADATA_PATH,
        DEVICE
    )

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        DEVICE
    )

    best_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc = trainer.train_one_epoch(
            train_loader
        )

        val_loss, val_acc = trainer.validate(
            val_loader
        )

        print("-" * 60)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        print(f"Train Loss : {train_loss:.4f}")

        print(f"Train Accuracy : {train_acc:.2f}%")

        print(f"Validation Loss : {val_loss:.4f}")

        print(f"Validation Accuracy : {val_acc:.2f}%")

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            torch.save(
                model.state_dict(),
                MODEL_DIR / "best_model.pth"
            )

            print("Best model saved.")

    print("\nTraining Completed.")


if __name__ == "__main__":
    main()