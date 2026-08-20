from src.preprocessing.create_dataloader import (
    create_dataloaders
)


train_loader, val_loader, test_loader = (
    create_dataloaders(
        batch_size=8
    )
)


print("\nDataloader test successful.")

print(
    "Training batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)

print(
    "Test batches:",
    len(test_loader)
)


images, labels = next(
    iter(train_loader)
)


print(
    "\nImage batch shape:",
    images.shape
)

print(
    "Label batch shape:",
    labels.shape
)

print(
    "Labels:",
    labels.tolist()
)