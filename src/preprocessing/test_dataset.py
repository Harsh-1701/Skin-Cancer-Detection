import pandas as pd

from src.preprocessing.skin_dataset import SkinCancerDataset
from src.preprocessing.transforms import val_transform
from src.utils.config import METADATA_PATH


def main():

    df = pd.read_csv(METADATA_PATH)

    dataset = SkinCancerDataset(
        dataframe=df,
        transform=val_transform
    )

    image, label = dataset[0]

    print("Image Shape :", image.shape)
    print("Label :", label)


if __name__ == "__main__":
    main()