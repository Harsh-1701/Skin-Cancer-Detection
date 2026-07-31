import os
from pathlib import Path

import pandas as pd


class HAM10000DatasetLoader:
    """
    Loads and verifies the HAM10000 dataset.
    """

    def __init__(self, project_root):

        self.project_root = Path(project_root)

        self.metadata_path = (
            self.project_root
            / "dataset"
            / "metadata"
            / "HAM10000_metadata.csv"
        )

        self.image_dirs = [

            self.project_root
            / "dataset"
            / "images"
            / "HAM10000_images_part_1",

            self.project_root
            / "dataset"
            / "images"
            / "HAM10000_images_part_2"
        ]

        self.metadata = None

    def load_metadata(self):

        print("=" * 60)
        print("Loading metadata...")
        print("=" * 60)

        self.metadata = pd.read_csv(self.metadata_path)

        print(f"Total samples : {len(self.metadata)}")

        print("\nColumns:")

        print(self.metadata.columns.tolist())

        return self.metadata

    def verify_images(self):

        print("\nVerifying image files...")

        missing = []

        for image_id in self.metadata["image_id"]:

            found = False

            for folder in self.image_dirs:

                image_path = folder / f"{image_id}.jpg"

                if image_path.exists():
                    found = True
                    break

            if not found:
                missing.append(image_id)

        print(f"Missing images : {len(missing)}")

        if len(missing) == 0:
            print("Dataset verification PASSED")

        else:

            print("Some images are missing.")

        return missing


if __name__ == "__main__":

    # Automatically detect the project root
    project_root = Path(__file__).resolve().parents[2]

    print(f"Project Root: {project_root}")

    loader = HAM10000DatasetLoader(project_root)

    loader.load_metadata()

    loader.verify_images()