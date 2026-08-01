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

# Class Labels
LABEL_MAP = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}

INDEX_TO_LABEL = {
    value: key
    for key, value in LABEL_MAP.items()
}

# Benign / Malignant Mapping
CLASS_INFO = {
    "akiec": "Malignant",
    "bcc": "Malignant",
    "mel": "Malignant",
    "bkl": "Benign",
    "df": "Benign",
    "nv": "Benign",
    "vasc": "Benign",
}

DISEASE_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

DISEASE_DESCRIPTION = {
    "akiec": "Potentially malignant skin lesion.",
    "bcc": "Common skin cancer with slow growth.",
    "bkl": "Benign skin lesion.",
    "df": "Benign fibrous skin lesion.",
    "mel": "Highly aggressive skin cancer.",
    "nv": "Common benign mole.",
    "vasc": "Benign vascular lesion.",
}