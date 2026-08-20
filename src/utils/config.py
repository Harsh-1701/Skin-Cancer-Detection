from pathlib import Path

# -------------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# -------------------------------------------------------
# DATASET PATHS
# -------------------------------------------------------

DATASET_DIR = PROJECT_ROOT / "dataset"

HAM_IMAGE_DIRS = [
    DATASET_DIR / "images" / "HAM10000_images_part_1",
    DATASET_DIR / "images" / "HAM10000_images_part_2",
]

ISIC_IMAGE_DIR = (
    DATASET_DIR
    / "ISIC2019"
    / "images"
)

UNKNOWN_IMAGE_DIR = (
    DATASET_DIR
    / "train"
    / "unknown"
)

COMBINED_DATASET_PATH = (
    DATASET_DIR
    / "combined_dataset.csv"
)


# -------------------------------------------------------
# OUTPUT DIRECTORIES
# -------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"


# -------------------------------------------------------
# IMAGE SETTINGS
# -------------------------------------------------------

IMAGE_SIZE = (224, 224)


# -------------------------------------------------------
# RANDOM SEED
# -------------------------------------------------------

RANDOM_SEED = 42


# -------------------------------------------------------
# CLASS LABELS
# -------------------------------------------------------

LABEL_MAP = {

    "akiec": 0,

    "bcc": 1,

    "bkl": 2,

    "df": 3,

    "mel": 4,

    "nv": 5,

    "vasc": 6,

    "scc": 7,

    "unknown": 8,

}


INDEX_TO_LABEL = {

    value: key

    for key, value in LABEL_MAP.items()

}


# -------------------------------------------------------
# BENIGN / MALIGNANT / UNKNOWN
# -------------------------------------------------------

CLASS_INFO = {

    "akiec": "Malignant",

    "bcc": "Malignant",

    "bkl": "Benign",

    "df": "Benign",

    "mel": "Malignant",

    "nv": "Benign",

    "vasc": "Benign",

    "scc": "Malignant",

    "unknown": "Unknown",

}


# -------------------------------------------------------
# DISEASE NAMES
# -------------------------------------------------------

DISEASE_NAMES = {

    "akiec": "Actinic Keratoses",

    "bcc": "Basal Cell Carcinoma",

    "bkl": "Benign Keratosis",

    "df": "Dermatofibroma",

    "mel": "Melanoma",

    "nv": "Melanocytic Nevus",

    "vasc": "Vascular Lesion",

    "scc": "Squamous Cell Carcinoma",

    "unknown": "Unknown / Non-Skin Image",

}


# -------------------------------------------------------
# DISEASE DESCRIPTIONS
# -------------------------------------------------------

DISEASE_DESCRIPTION = {

    "akiec":
        "Actinic keratoses are rough, scaly skin lesions caused by long-term sun exposure and may require clinical evaluation.",

    "bcc":
        "Basal cell carcinoma is a common type of skin cancer that generally grows slowly.",

    "bkl":
        "Benign keratosis is a non-cancerous skin lesion.",

    "df":
        "Dermatofibroma is a common benign fibrous skin lesion.",

    "mel":
        "Melanoma is a potentially aggressive form of skin cancer that requires professional medical evaluation.",

    "nv":
        "Melanocytic nevus is a common benign mole or pigmented skin lesion.",

    "vasc":
        "Vascular lesions are skin abnormalities involving blood vessels and may be benign.",

    "scc":
        "Squamous cell carcinoma is a type of skin cancer that requires professional medical evaluation.",

    "unknown":
        "The image does not correspond to one of the supported skin lesion categories. It may be a non-skin image or an image outside the model's supported categories.",

}