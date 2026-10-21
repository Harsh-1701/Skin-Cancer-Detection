from pathlib import Path

import cv2
from torch.utils.data import Dataset

from src.utils.config import DATASET_DIR, LABEL_MAP


class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None, image_index=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_index = (
            self._build_image_index()
            if image_index is None
            else image_index
        )

    @staticmethod
    def _build_image_index():
        print("\nBuilding image path index...")

        image_index = {
            "HAM10000": {},
            "ISIC2019": {},
            "IMAGENETTE": {},
        }

        ham_folders = [
            DATASET_DIR / "images" / "HAM10000_images_part_1",
            DATASET_DIR / "images" / "HAM10000_images_part_2",
        ]

        for folder in ham_folders:
            if folder.exists():
                for path in folder.iterdir():
                    if path.is_file():
                        image_index["HAM10000"][path.name] = path

        isic_root = DATASET_DIR / "ISIC2019" / "images"
        if isic_root.exists():
            for path in isic_root.rglob("*"):
                if path.is_file():
                    image_index["ISIC2019"][path.name] = path

        unknown_root = DATASET_DIR / "train" / "unknown"
        if unknown_root.exists():
            for path in unknown_root.iterdir():
                if path.is_file():
                    image_index["IMAGENETTE"][path.name] = path

        for source, paths in image_index.items():
            print(f"{source} images indexed: {len(paths)}")

        return image_index

    def __len__(self):
        return len(self.dataframe)

    def _find_image(self, row):
        image_name = row["image_name"]
        source = row["source"]

        try:
            return self.image_index[source][image_name]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Image not found: {image_name} (source: {source})"
            ) from exc

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self._find_image(row)

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = LABEL_MAP[row["dx"]]
        sample_id = f"{row['source']}::{row['image_name']}"

        return image, label, sample_id