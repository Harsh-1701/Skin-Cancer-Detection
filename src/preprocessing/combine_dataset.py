from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HAM_METADATA = PROJECT_ROOT / "dataset" / "metadata" / "HAM10000_metadata.csv"

ISIC_METADATA = PROJECT_ROOT / "dataset" / "ISIC2019" / "ISIC_2019_Training_Metadata.csv"

ISIC_GT = PROJECT_ROOT / "dataset" / "ISIC2019" / "ISIC_2019_Training_GroundTruth.csv"

ham = pd.read_csv(HAM_METADATA)

meta = pd.read_csv(ISIC_METADATA)

gt = pd.read_csv(ISIC_GT)

isic = meta.merge(gt, on="image")

label_columns = [
    "MEL",
    "NV",
    "BCC",
    "AK",
    "BKL",
    "DF",
    "VASC",
    "SCC",
]

isic["dx"] = isic[label_columns].idxmax(axis=1)

mapping = {
    "AK": "akiec",
    "BCC": "bcc",
    "BKL": "bkl",
    "DF": "df",
    "MEL": "mel",
    "NV": "nv",
    "VASC": "vasc",
    "SCC": "scc",
}

isic["dx"] = isic["dx"].map(mapping)

print("\nHAM Distribution\n")

print(ham["dx"].value_counts())

print("\nISIC Distribution\n")

print(isic["dx"].value_counts())