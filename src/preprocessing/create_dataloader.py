import pandas as pd

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from src.preprocessing.skin_dataset import SkinCancerDataset
from src.preprocessing.transforms import train_transform, val_transform
from src.utils.config import METADATA_PATH


def create_dataloaders(batch_size=32):

    df = pd.read_csv(METADATA_PATH)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["dx"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["dx"]
    )

    train_dataset = SkinCancerDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = SkinCancerDataset(
        val_df,
        transform=val_transform
    )

    test_dataset = SkinCancerDataset(
        test_df,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader