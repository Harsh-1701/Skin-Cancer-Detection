from pathlib import Path

import cv2
from torch.utils.data import Dataset

from src.utils.config import (
    DATASET_DIR,
    LABEL_MAP,
)


class SkinCancerDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None,
        image_index=None,
    ):

        self.dataframe = dataframe.reset_index(drop=True)

        self.transform = transform

        # ---------------------------------------------------
        # IMAGE PATH INDEX
        # ---------------------------------------------------

        if image_index is None:
            self.image_index = self._build_image_index()
        else:
            self.image_index = image_index

    # -------------------------------------------------------
    # BUILD IMAGE INDEX
    # -------------------------------------------------------

    @staticmethod
    def _build_image_index():

        print("\nBuilding image path index...")

        image_index = {
            "HAM10000": {},
            "ISIC2019": {},
            "IMAGENETTE": {},
        }

        # ---------------------------------------------------
        # HAM10000
        # ---------------------------------------------------

        ham_folders = [
            DATASET_DIR
            / "images"
            / "HAM10000_images_part_1",

            DATASET_DIR
            / "images"
            / "HAM10000_images_part_2",
        ]

        for folder in ham_folders:

            if not folder.exists():
                continue

            for image_path in folder.iterdir():

                if image_path.is_file():

                    image_index["HAM10000"][
                        image_path.name
                    ] = image_path

        # ---------------------------------------------------
        # ISIC2019
        # ---------------------------------------------------

        isic_root = (
            DATASET_DIR
            / "ISIC2019"
            / "images"
        )

        if isic_root.exists():

            for image_path in isic_root.rglob("*"):

                if image_path.is_file():

                    image_index["ISIC2019"][
                        image_path.name
                    ] = image_path

        # ---------------------------------------------------
        # IMAGENETTE UNKNOWN
        # ---------------------------------------------------

        unknown_root = (
            DATASET_DIR
            / "train"
            / "unknown"
        )

        if unknown_root.exists():

            for image_path in unknown_root.iterdir():

                if image_path.is_file():

                    image_index["IMAGENETTE"][
                        image_path.name
                    ] = image_path

        print(
            "HAM10000 images indexed :",
            len(image_index["HAM10000"])
        )

        print(
            "ISIC2019 images indexed :",
            len(image_index["ISIC2019"])
        )

        print(
            "Unknown images indexed  :",
            len(image_index["IMAGENETTE"])
        )

        return image_index

    # -------------------------------------------------------
    # LENGTH
    # -------------------------------------------------------

    def __len__(self):

        return len(self.dataframe)

    # -------------------------------------------------------
    # FIND IMAGE
    # -------------------------------------------------------

    def _find_image(self, row):

        image_name = row["image_name"]

        source = row["source"]

        try:

            return self.image_index[source][image_name]

        except KeyError:

            raise FileNotFoundError(
                f"Image not found: {image_name} "
                f"(source: {source})"
            )

    # -------------------------------------------------------
    # GET ITEM
    # -------------------------------------------------------

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