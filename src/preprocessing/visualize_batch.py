import matplotlib.pyplot as plt

from src.preprocessing.create_dataloader import create_dataloaders
from src.utils.config import IDX_TO_LABEL


def main():

    train_loader, _, _ = create_dataloaders()

    images, labels = next(iter(train_loader))

    plt.figure(figsize=(12, 12))

    for i in range(16):

        plt.subplot(4, 4, i + 1)

        image = images[i].permute(1, 2, 0).numpy()

        image = (image - image.min()) / (image.max() - image.min())

        plt.imshow(image)

        plt.title(IDX_TO_LABEL[int(labels[i])])

        plt.axis("off")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()