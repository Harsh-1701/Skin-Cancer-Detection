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


# -------------------------------------------------------
# DATALOADER PERFORMANCE SETTINGS
# -------------------------------------------------------

TRAIN_NUM_WORKERS = 2

VAL_NUM_WORKERS = 0

PIN_MEMORY = True

PERSISTENT_WORKERS = True

PREFETCH_FACTOR = 2


def create_dataloaders(
    batch_size=4
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
    # ---------------------------------------------------

    train_df, temp_df = train_test_split(

        df,

        test_size=0.30,

        random_state=RANDOM_SEED,

        stratify=df["dx"],
    )

    # ---------------------------------------------------
    # VALIDATION / TEST SPLIT
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
    # BUILD IMAGE INDEX ONCE
    # ---------------------------------------------------

    print(
        "\nBuilding image path index..."
    )

    image_index = (
        SkinCancerDataset
        ._build_image_index()
    )

    # ---------------------------------------------------
    # CREATE DATASETS
    # ---------------------------------------------------

    train_dataset = SkinCancerDataset(

        train_df,

        transform=train_transform,

        image_index=image_index,
    )

    val_dataset = SkinCancerDataset(

        val_df,

        transform=val_transform,

        image_index=image_index,
    )

    test_dataset = SkinCancerDataset(

        test_df,

        transform=val_transform,

        image_index=image_index,
    )

    # ---------------------------------------------------
    # TRAIN DATALOADER
    # ---------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=TRAIN_NUM_WORKERS,

        pin_memory=PIN_MEMORY,

        persistent_workers=PERSISTENT_WORKERS,

        prefetch_factor=PREFETCH_FACTOR,
    )

    # ---------------------------------------------------
    # VALIDATION DATALOADER
    # ---------------------------------------------------

    val_loader = DataLoader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=VAL_NUM_WORKERS,

        pin_memory=PIN_MEMORY,
    )

    # ---------------------------------------------------
    # TEST DATALOADER
    # ---------------------------------------------------

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=VAL_NUM_WORKERS,

        pin_memory=PIN_MEMORY,
    )

    # ---------------------------------------------------
    # RETURN
    # ---------------------------------------------------

    return (
        train_loader,
        val_loader,
        test_loader,
    )