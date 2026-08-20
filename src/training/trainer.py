import torch


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device


    # ---------------------------------------------------
    # TRAIN ONE EPOCH
    # ---------------------------------------------------

    def train_one_epoch(
        self,
        dataloader
    ):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for images, labels in dataloader:

            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            # -------------------------------------------
            # CLEAR GRADIENTS
            # -------------------------------------------

            self.optimizer.zero_grad()

            # -------------------------------------------
            # FORWARD PASS
            # -------------------------------------------

            outputs = self.model(
                images
            )

            # -------------------------------------------
            # LOSS
            # -------------------------------------------

            loss = self.criterion(
                outputs,
                labels
            )

            # -------------------------------------------
            # BACKPROPAGATION
            # -------------------------------------------

            loss.backward()

            # -------------------------------------------
            # UPDATE MODEL
            # -------------------------------------------

            self.optimizer.step()

            # -------------------------------------------
            # STATISTICS
            # -------------------------------------------

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
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
        dataloader
    ):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        for images, labels in dataloader:

            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            # -------------------------------------------
            # FORWARD PASS
            # -------------------------------------------

            outputs = self.model(
                images
            )

            # -------------------------------------------
            # LOSS
            # -------------------------------------------

            loss = self.criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            # -------------------------------------------
            # PREDICTIONS
            # -------------------------------------------

            _, predicted = torch.max(
                outputs,
                1
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