from pathlib import Path
import pandas as pd
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

HAM_METADATA = PROJECT_ROOT / "dataset" / "metadata" / "HAM10000_metadata.csv"

ISIC_METADATA = PROJECT_ROOT / "dataset" / "ISIC2019" / "ISIC_2019_Training_Metadata.csv"

ISIC_GT = PROJECT_ROOT / "dataset" / "ISIC2019" / "ISIC_2019_Training_GroundTruth.csv"

UNKNOWN_ROOT = PROJECT_ROOT / "dataset" / "train" / "unknown"

OUTPUT_CSV = PROJECT_ROOT / "dataset" / "combined_dataset.csv"

# ---------------------------------------------------
# HAM10000
# ---------------------------------------------------

ham = pd.read_csv(HAM_METADATA)

ham["source"] = "HAM10000"

ham["image_name"] = ham["image_id"] + ".jpg"

ham = ham[["image_name", "dx", "source"]]

print("HAM Loaded :", len(ham))

# ---------------------------------------------------
# ISIC
# ---------------------------------------------------

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

isic["dx"] = (
    isic[label_columns]
    .idxmax(axis=1)
    .map(mapping)
)

isic["image_name"] = isic["image"] + ".jpg"

isic["source"] = "ISIC2019"

isic = isic[
    ["image_name", "dx", "source"]
]

print("ISIC Loaded :", len(isic))

# ---------------------------------------------------
# UNKNOWN IMAGES
# ---------------------------------------------------

unknown_rows = []

for image_path in UNKNOWN_ROOT.glob("*"):

    if image_path.suffix.lower() not in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    ]:
        continue

    unknown_rows.append(
        {
            "image_name": image_path.name,
            "dx": "unknown",
            "source": "IMAGENETTE",
        }
    )

print("Unknown Images :", len(unknown_rows))

# ---------------------------------------------------
# SAMPLE UNKNOWN
# ---------------------------------------------------

random.seed(42)

if len(unknown_rows) > 6000:

    unknown_rows = random.sample(
        unknown_rows,
        6000,
    )

unknown = pd.DataFrame(
    unknown_rows
)

print(
    "Unknown Used :",
    len(unknown),
)

# ---------------------------------------------------
# COMBINE
# ---------------------------------------------------

combined = pd.concat(

    [

        ham,

        isic,

        unknown,

    ],

    ignore_index=True,

)

combined = combined.sample(

    frac=1,

    random_state=42,

)

combined.to_csv(

    OUTPUT_CSV,

    index=False,

)

print("=" * 60)

print("Combined Dataset Created")

print("=" * 60)

print(combined["dx"].value_counts())

print("=" * 60)

print("Saved to")

print(OUTPUT_CSV)