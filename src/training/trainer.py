import torch

from torch.amp import (
    autocast,
    GradScaler,
)


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        gradient_accumulation_steps=1,
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device

        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )

        self.scaler = GradScaler(
            "cuda",
            enabled=(
                device.type == "cuda"
            ),
        )


    # ---------------------------------------------------
    # TRAIN ONE EPOCH
    # ---------------------------------------------------

    def train_one_epoch(
        self,
        dataloader,
    ):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        self.optimizer.zero_grad(
            set_to_none=True
        )


        for batch_index, (
            images,
            labels
        ) in enumerate(dataloader):

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )


            # -------------------------------------------
            # FORWARD PASS
            # -------------------------------------------

            with autocast(
                device_type=self.device.type,
                enabled=(
                    self.device.type == "cuda"
                ),
            ):

                outputs = self.model(
                    images
                )

                loss = self.criterion(
                    outputs,
                    labels,
                )


            # -------------------------------------------
            # SAVE ORIGINAL LOSS
            # -------------------------------------------

            original_loss = loss.item()

            running_loss += (
                original_loss
            )


            # -------------------------------------------
            # GRADIENT ACCUMULATION
            # -------------------------------------------

            loss = (
                loss
                / self.gradient_accumulation_steps
            )


            self.scaler.scale(
                loss
            ).backward()


            # -------------------------------------------
            # OPTIMIZER STEP
            # -------------------------------------------

            should_update = (

                (batch_index + 1)
                % self.gradient_accumulation_steps
                == 0

                or

                (batch_index + 1)
                == len(dataloader)
            )


            if should_update:

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                self.optimizer.zero_grad(
                    set_to_none=True
                )


            # -------------------------------------------
            # STATISTICS
            # -------------------------------------------

            _, predicted = torch.max(
                outputs,
                1,
            )

            total += labels.size(0)

            correct += (
                predicted
                .eq(labels)
                .sum()
                .item()
            )


        average_loss = (
            running_loss
            / len(dataloader)
        )


        accuracy = (
            100.0
            * correct
            / total
        )


        return (
            average_loss,
            accuracy,
        )


    # ---------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------

    @torch.no_grad()
    def validate(
        self,
        dataloader,
    ):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in dataloader:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )


            # -------------------------------------------
            # FORWARD PASS
            # -------------------------------------------

            with autocast(
                device_type=self.device.type,
                enabled=(
                    self.device.type == "cuda"
                ),
            ):

                outputs = self.model(
                    images
                )


            # -------------------------------------------
            # LOSS
            # -------------------------------------------

            loss = self.criterion(
                outputs.float(),
                labels,
            )


            running_loss += (
                loss.item()
            )


            # -------------------------------------------
            # PREDICTIONS
            # -------------------------------------------

            _, predicted = torch.max(
                outputs,
                1,
            )


            total += labels.size(0)


            correct += (
                predicted
                .eq(labels)
                .sum()
                .item()
            )


        average_loss = (
            running_loss
            / len(dataloader)
        )


        accuracy = (
            100.0
            * correct
            / total
        )


        return (
            average_loss,
            accuracy,
        )