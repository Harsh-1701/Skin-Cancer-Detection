import csv
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.efficientnet import build_model

from src.preprocessing.create_dataloader import (
    create_dataloaders
)

from src.training.loss import (
    create_weighted_loss
)

from src.training.train_config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

from src.training.trainer import Trainer

from src.utils.config import (
    COMBINED_DATASET_PATH,
    MODEL_DIR,
    LOG_DIR,
)


# ---------------------------------------------------
# OUTPUT DIRECTORIES
# ---------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------
# TRAINING LOG
# ---------------------------------------------------

LOG_FILE = (
    LOG_DIR
    / "training_log_b0_improved.csv"
)


# ---------------------------------------------------
# CHECKPOINT FUNCTION
# ---------------------------------------------------

def save_checkpoint(
    filename,
    epoch,
    model,
    optimizer,
    scheduler,
    val_acc,
):

    torch.save(

        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "validation_accuracy":
                val_acc,
        },

        MODEL_DIR / filename,
    )


# ---------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------

def main():

    print("=" * 60)
    print("Skin Cancer Detection System - V2")
    print("Improved EfficientNet-B0 Experiment")
    print("=" * 60)

    print(
        "\nDevice:",
        DEVICE
    )

    print(
        "Number of classes:",
        9
    )

    # ------------------------------------------------
    # DATA LOADERS
    # ------------------------------------------------

    train_loader, val_loader, _ = (
        create_dataloaders(
            batch_size=BATCH_SIZE
        )
    )

    # ------------------------------------------------
    # MODEL
    # ------------------------------------------------

    model = build_model(
        num_classes=9
    )

    # ------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------

    optimizer = AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )

    # ------------------------------------------------
    # COSINE LEARNING-RATE SCHEDULER
    # ------------------------------------------------

    scheduler = CosineAnnealingLR(

        optimizer,

        T_max=NUM_EPOCHS,

        eta_min=1e-6,
    )

    # ------------------------------------------------
    # LOSS
    # ------------------------------------------------

    criterion = create_weighted_loss(

        COMBINED_DATASET_PATH,

        DEVICE,
    )

    # ------------------------------------------------
    # TRAINER
    # ------------------------------------------------

    trainer = Trainer(

        model,

        optimizer,

        criterion,

        DEVICE,
    )

    best_accuracy = 0.0

    # ------------------------------------------------
    # LOG
    # ------------------------------------------------

    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Epoch",
                "Learning Rate",
                "Train Loss",
                "Train Accuracy",
                "Validation Loss",
                "Validation Accuracy",
            ]
        )

        # --------------------------------------------
        # EPOCH LOOP
        # --------------------------------------------

        for epoch in range(NUM_EPOCHS):

            train_loss, train_acc = (
                trainer.train_one_epoch(
                    train_loader
                )
            )

            val_loss, val_acc = (
                trainer.validate(
                    val_loader
                )
            )

            current_lr = (
                optimizer.param_groups[0]["lr"]
            )

            print(
                "\n" + "-" * 60
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{NUM_EPOCHS}"
            )

            print(
                f"Learning Rate : "
                f"{current_lr:.8f}"
            )

            print(
                f"Train Loss : "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Accuracy : "
                f"{train_acc:.2f}%"
            )

            print(
                f"Validation Loss : "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation Accuracy : "
                f"{val_acc:.2f}%"
            )

            # ----------------------------------------
            # SAVE LOG
            # ----------------------------------------

            writer.writerow(
                [
                    epoch + 1,
                    current_lr,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                ]
            )

            file.flush()

            # ----------------------------------------
            # CHECKPOINT
            # ----------------------------------------

            save_checkpoint(

                f"b0_improved_epoch_"
                f"{epoch + 1:02d}.pth",

                epoch + 1,

                model,

                optimizer,

                scheduler,

                val_acc,
            )

            # ----------------------------------------
            # BEST MODEL
            # ----------------------------------------

            if val_acc > best_accuracy:

                best_accuracy = val_acc

                save_checkpoint(

                    "best_model_b0_improved.pth",

                    epoch + 1,

                    model,

                    optimizer,

                    scheduler,

                    val_acc,
                )

                print(
                    f"\nBest improved B0 model saved!"
                    f" Validation Accuracy: "
                    f"{val_acc:.2f}%"
                )

            # ----------------------------------------
            # UPDATE LEARNING RATE
            # ----------------------------------------

            scheduler.step()

    # ------------------------------------------------
    # COMPLETE
    # ------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "Improved B0 Training Completed."
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        f"Model saved in: "
        f"{MODEL_DIR}"
    )

    print(
        f"Training log saved in: "
        f"{LOG_FILE}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()