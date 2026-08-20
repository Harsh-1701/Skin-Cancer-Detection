import pandas as pd

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from src.preprocessing.skin_dataset import SkinCancerDataset

from src.preprocessing.transforms import (
    train_transform,
    val_transform,
)

from src.utils.config import (
    COMBINED_DATASET_PATH,
    RANDOM_SEED,
)


def create_dataloaders(
    batch_size=32
):

    # ---------------------------------------------------
    # LOAD COMBINED DATASET
    # ---------------------------------------------------

    df = pd.read_csv(
        COMBINED_DATASET_PATH
    )

    print(
        "Total images:",
        len(df)
    )

    print(
        "\nClass distribution:"
    )

    print(
        df["dx"].value_counts()
    )

    # ---------------------------------------------------
    # TRAIN / TEMP SPLIT
    # 70% TRAIN
    # 30% TEMPORARY
    # ---------------------------------------------------

    train_df, temp_df = train_test_split(

        df,

        test_size=0.30,

        random_state=RANDOM_SEED,

        stratify=df["dx"],
    )

    # ---------------------------------------------------
    # VALIDATION / TEST SPLIT
    # 15% VALIDATION
    # 15% TEST
    # ---------------------------------------------------

    val_df, test_df = train_test_split(

        temp_df,

        test_size=0.50,

        random_state=RANDOM_SEED,

        stratify=temp_df["dx"],
    )

    print(
        "\nTrain:",
        len(train_df)
    )

    print(
        "Validation:",
        len(val_df)
    )

    print(
        "Test:",
        len(test_df)
    )

    # ---------------------------------------------------
    # CREATE DATASETS
    # ---------------------------------------------------

    train_dataset = SkinCancerDataset(

        train_df,

        transform=train_transform,
    )

    val_dataset = SkinCancerDataset(

        val_df,

        transform=val_transform,
    )

    test_dataset = SkinCancerDataset(

        test_df,

        transform=val_transform,
    )

    # ---------------------------------------------------
    # CREATE DATALOADERS
    # ---------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=0,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=0,
    )

    # ---------------------------------------------------
    # RETURN LOADERS
    # ---------------------------------------------------

    return (
        train_loader,
        val_loader,
        test_loader,
    )