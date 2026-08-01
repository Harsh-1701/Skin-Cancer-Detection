import csv
import torch

from pathlib import Path
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
    LOG_DIR,
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "training_log.csv"


def save_checkpoint(filename, epoch, model, optimizer, val_acc):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_accuracy": val_acc,
        },
        MODEL_DIR / filename,
    )


def main():

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = build_model()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = create_weighted_loss(
        METADATA_PATH,
        DEVICE,
    )

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        DEVICE,
    )

    best_accuracy = 0.0

    with open(LOG_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Epoch",
                "Train Loss",
                "Train Accuracy",
                "Validation Loss",
                "Validation Accuracy",
            ]
        )

        for epoch in range(NUM_EPOCHS):

            train_loss, train_acc = trainer.train_one_epoch(
                train_loader
            )

            val_loss, val_acc = trainer.validate(
                val_loader
            )

            print("-" * 60)

            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

            print(f"Train Loss : {train_loss:.4f}")

            print(f"Train Accuracy : {train_acc:.2f}%")

            print(f"Validation Loss : {val_loss:.4f}")

            print(f"Validation Accuracy : {val_acc:.2f}%")

            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                ]
            )

            save_checkpoint(
                f"checkpoint_epoch_{epoch+1:02d}.pth",
                epoch + 1,
                model,
                optimizer,
                val_acc,
            )

            save_checkpoint(
                "last_model.pth",
                epoch + 1,
                model,
                optimizer,
                val_acc,
            )

            if val_acc > best_accuracy:

                best_accuracy = val_acc

                save_checkpoint(
                    "best_model.pth",
                    epoch + 1,
                    model,
                    optimizer,
                    val_acc,
                )

                print(
                    f"Best model saved! Validation Accuracy: {val_acc:.2f}%"
                )

    print("\nTraining Completed.")


if __name__ == "__main__":
    main()