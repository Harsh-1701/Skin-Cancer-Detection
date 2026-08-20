from pathlib import Path

import cv2
from torch.utils.data import Dataset

from src.utils.config import (
    DATASET_DIR,
    LABEL_MAP,
)


class SkinCancerDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def _find_image(self, row):

        image_name = row["image_name"]
        source = row["source"]

        # ---------------------------------------------------
        # HAM10000
        # ---------------------------------------------------

        if source == "HAM10000":

            ham_folders = [
                DATASET_DIR / "images" / "HAM10000_images_part_1",
                DATASET_DIR / "images" / "HAM10000_images_part_2",
            ]

            for folder in ham_folders:

                image_path = folder / image_name

                if image_path.exists():

                    return image_path

        # ---------------------------------------------------
        # ISIC2019
        # ---------------------------------------------------

        elif source == "ISIC2019":

            isic_root = (
                DATASET_DIR
                / "ISIC2019"
                / "images"
            )

            for folder in isic_root.iterdir():

                if folder.is_dir():

                    image_path = folder / image_name

                    if image_path.exists():

                        return image_path

        # ---------------------------------------------------
        # IMAGENETTE UNKNOWN
        # ---------------------------------------------------

        elif source == "IMAGENETTE":

            image_path = (
                DATASET_DIR
                / "train"
                / "unknown"
                / image_name
            )

            if image_path.exists():

                return image_path

        # ---------------------------------------------------
        # IMAGE NOT FOUND
        # ---------------------------------------------------

        raise FileNotFoundError(
            f"Image not found: {image_name} "
            f"(source: {source})"
        )

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = self._find_image(row)

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        label = LABEL_MAP[row["dx"]]

        if self.transform:

            image = self.transform(
                image=image
            )["image"]

        return image, label