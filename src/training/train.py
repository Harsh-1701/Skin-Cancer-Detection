import csv
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

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
    BACKBONE_LR,
    CLASSIFIER_LR,
    WEIGHT_DECAY,
    WARMUP_EPOCHS,
    MIN_LR,
    LABEL_SMOOTHING,
    CLASS_WEIGHT_POWER,
    GRADIENT_ACCUMULATION_STEPS,
)

from src.training.trainer import Trainer

from src.utils.config import (
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
    / "training_log_efficientnet_b4.csv"
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
    print("EfficientNet-B4 Experiment")
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

    classifier_parameters = list(
        model.classifier.parameters()
    )

    classifier_parameter_ids = {
        id(parameter)
        for parameter in classifier_parameters
    }

    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in classifier_parameter_ids
    ]

    optimizer = AdamW(

        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LR,
            },
            {
                "params": classifier_parameters,
                "lr": CLASSIFIER_LR,
            },
        ],

        weight_decay=WEIGHT_DECAY,
    )

    # ------------------------------------------------
    # WARMUP + COSINE LEARNING-RATE SCHEDULER
    # ------------------------------------------------

    warmup_scheduler = LinearLR(

        optimizer,

        start_factor=1 / WARMUP_EPOCHS,

        end_factor=1.0,

        total_iters=WARMUP_EPOCHS,
    )

    cosine_scheduler = CosineAnnealingLR(

        optimizer,

        T_max=NUM_EPOCHS - WARMUP_EPOCHS,

        eta_min=MIN_LR,
    )

    scheduler = SequentialLR(

        optimizer,

        schedulers=[
            warmup_scheduler,
            cosine_scheduler,
        ],

        milestones=[WARMUP_EPOCHS],
    )

    # ------------------------------------------------
    # LOSS
    # ------------------------------------------------

    criterion = create_weighted_loss(

        train_loader.dataset.dataframe["dx"],

        DEVICE,

        class_weight_power=CLASS_WEIGHT_POWER,

        label_smoothing=LABEL_SMOOTHING,
    )

    # ------------------------------------------------
    # TRAINER
    # ------------------------------------------------

    trainer = Trainer(

        model,

        optimizer,

        criterion,

        DEVICE,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),
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
                "Backbone Learning Rate",
                "Classifier Learning Rate",
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

            backbone_lr = (
                optimizer.param_groups[0]["lr"]
            )

            classifier_lr = (
                optimizer.param_groups[1]["lr"]
            )

            print(
                "\n" + "-" * 60
            )

            print(
                f"Epoch "
                f"{epoch + 1}/{NUM_EPOCHS}"
            )

            print(
                f"Backbone Learning Rate : "
                f"{backbone_lr:.8f}"
            )

            print(
                f"Classifier Learning Rate : "
                f"{classifier_lr:.8f}"
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
                    backbone_lr,
                    classifier_lr,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                ]
            )

            file.flush()

            # ----------------------------------------
            # UPDATE LEARNING RATE
            # ----------------------------------------

            scheduler.step()

            # ----------------------------------------
            # CHECKPOINT
            # ----------------------------------------

            save_checkpoint(

                f"efficientnet_b4_epoch_"
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

                    "best_model_efficientnet_b4.pth",

                    epoch + 1,

                    model,

                    optimizer,

                    scheduler,

                    val_acc,
                )

                print(
                    f"\nBest EfficientNet-B4 model saved!"
                    f" Validation Accuracy: "
                    f"{val_acc:.2f}%"
                )

    # ------------------------------------------------
    # COMPLETE
    # ------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "EfficientNet-B4 Training Completed."
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
