import random
import shutil
from pathlib import Path

random.seed(42)

SOURCE = Path(r"D:\Datasets\imagenette2\imagenette2\train")
DESTINATION = Path(r"D:\Skin Cancer\dataset\train\unknown")

DESTINATION.mkdir(parents=True, exist_ok=True)

images = []

for folder in SOURCE.iterdir():

    if folder.is_dir():

        images.extend(folder.glob("*.JPEG"))

print(f"Found {len(images)} images.")

selected = random.sample(images, 1000)

for image in selected:

    shutil.copy(image, DESTINATION / image.name)

print("Done.")
print(f"Copied {len(selected)} images.")