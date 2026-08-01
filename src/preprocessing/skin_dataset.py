import cv2
import pandas as pd
from torch.utils.data import Dataset

from src.utils.config import IMAGE_DIRS, LABEL_MAP


class SkinCancerDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def _find_image(self, image_id):

        for folder in IMAGE_DIRS:

            image_path = folder / f"{image_id}.jpg"

            if image_path.exists():
                return image_path

        raise FileNotFoundError(f"{image_id}.jpg not found")

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = self._find_image(row["image_id"])

        image = cv2.imread(str(image_path))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = LABEL_MAP[row["dx"]]

        if self.transform:

            image = self.transform(image=image)["image"]

        return image, label