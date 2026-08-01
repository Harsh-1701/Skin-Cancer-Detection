from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset
DATASET_DIR = PROJECT_ROOT / "dataset"

IMAGE_DIRS = [
    DATASET_DIR / "images" / "HAM10000_images_part_1",
    DATASET_DIR / "images" / "HAM10000_images_part_2",
]

METADATA_PATH = (
    DATASET_DIR
    / "metadata"
    / "HAM10000_metadata.csv"
)

# Outputs
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# Image Size
IMAGE_SIZE = (224, 224)

# Random Seed
RANDOM_SEED = 42

# Diagnosis Labels
LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6
}

IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

BENIGN_CLASSES = {
    "nv",
    "bkl",
    "bcc",
    "df",
    "vasc"
}

MALIGNANT_CLASSES = {
    "mel",
    "akiec"
}
