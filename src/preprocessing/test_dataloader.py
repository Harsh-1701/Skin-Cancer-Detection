from src.preprocessing.create_dataloader import create_dataloaders


def main():

    train_loader, val_loader, test_loader = create_dataloaders()

    print("=" * 50)

    print("Training batches :", len(train_loader))

    print("Validation batches :", len(val_loader))

    print("Testing batches :", len(test_loader))

    print("=" * 50)

    images, labels = next(iter(train_loader))

    print("Image Batch Shape :", images.shape)

    print("Label Batch Shape :", labels.shape)


if __name__ == "__main__":
    main()