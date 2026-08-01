import torch


class Trainer:

    def __init__(self, model, optimizer, criterion, device):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, dataloader):

        self.model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

        loss = running_loss / len(dataloader)

        accuracy = 100 * correct / total

        return loss, accuracy

    @torch.no_grad()
    def validate(self, dataloader):

        self.model.eval()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

        loss = running_loss / len(dataloader)

        accuracy = 100 * correct / total

        return loss, accuracy