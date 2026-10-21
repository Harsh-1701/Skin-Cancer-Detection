from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import timm

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.preprocessing.skin_dataset import SkinCancerDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.training.train_config import DEVICE
from src.utils.config import (
    COMBINED_DATASET_PATH,
    RANDOM_SEED,
    INDEX_TO_LABEL,
)


B0_CHECKPOINT = Path("outputs/models/best_model_b0_final.pth")
B4_CHECKPOINT = Path("outputs/models/best_model_efficientnet_b4.pth")
OUTPUT_DIR = Path("outputs/paper2")

BATCH_SIZE_B0 = 16
BATCH_SIZE_B4 = 4
NUM_CLASSES = 9
SEED = 42

CLASS_NAMES = [INDEX_TO_LABEL[i] for i in range(NUM_CLASSES)]


class IdentifiedDataset(Dataset):
    """Adds stable source/filename IDs without changing the shared dataset."""

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.dataframe = base_dataset.dataframe

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]

        if len(item) == 3:
            image, label, _ = item
        else:
            image, label = item

        row = self.dataframe.iloc[index]
        sample_id = f"{row['source']}::{row['image_name']}"

        return image, label, sample_id


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

def make_eval_transform(image_size):
    """Deterministic evaluation transform for a specific model."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(),
        ToTensorV2(),
    ])


def create_paper2_loaders(batch_size, transform):
    """Create deterministic, leakage-resistant splits from unique image names."""
    from pathlib import Path

    manifest_path = Path("outputs/paper2/deduplicated_dataset_manifest.csv")
    split_path = Path("outputs/paper2/split_manifest_deduplicated.csv")

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Deduplicated manifest not found: {manifest_path}"
        )

    df = pd.read_csv(manifest_path)

    if df["image_name"].duplicated().any():
        raise ValueError("Duplicate image names remain in the evaluation manifest.")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df["dx"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["dx"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    split_manifest = pd.concat(
        [train_df, val_df, test_df], ignore_index=True
    )
    split_manifest.to_csv(split_path, index=False)

    # Verify that all split assignments are unique and exhaustive.
    assert split_manifest["image_name"].is_unique
    assert len(split_manifest) == len(df)

    image_index = SkinCancerDataset._build_image_index()

    loaders = {}
    for name, frame in (("val", val_df), ("test", test_df)):
        base = SkinCancerDataset(
            frame.drop(columns=["split"], errors="ignore"),
            transform=transform,
            image_index=image_index,
        )
        dataset = IdentifiedDataset(base)

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    print(
        f"Deduplicated split sizes: train={len(train_df)}, "
        f"validation={len(val_df)}, test={len(test_df)}"
    )
    print(f"Saved split manifest: {split_path}")

    return loaders


def load_model(model_name, checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} does not contain model_state_dict"
        )

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=NUM_CLASSES,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Loaded {model_name}: {checkpoint_path}")
    return model


def collect_predictions(model, loader, model_name, split_name):
    all_ids, all_labels, all_probs = [], [], []

    print(f"Predicting {split_name}: {model_name}")

    with torch.inference_mode():
        for images, labels, sample_ids in loader:
            images = images.to(DEVICE, non_blocking=True)
            probabilities = torch.softmax(model(images), dim=1)

            all_ids.extend(list(sample_ids))
            all_labels.extend(labels.numpy().tolist())
            all_probs.append(probabilities.cpu().numpy())

    return {
        "ids": np.asarray(all_ids),
        "y": np.asarray(all_labels, dtype=np.int64),
        "probs": np.concatenate(all_probs, axis=0),
    }


def verify_pairing(b0, b4, split_name):
    if not np.array_equal(b0["ids"], b4["ids"]):
        raise RuntimeError(f"{split_name}: B0/B4 sample IDs do not match.")

    if not np.array_equal(b0["y"], b4["y"]):
        raise RuntimeError(f"{split_name}: B0/B4 ground-truth labels do not match.")


def error_metrics(y_error, risk_score):
    """Higher score means greater predicted risk of classification error."""
    y_error = np.asarray(y_error, dtype=np.int32)
    risk_score = np.asarray(risk_score, dtype=np.float64)

    result = {
        "n_samples": int(len(y_error)),
        "errors": int(y_error.sum()),
        "error_rate": float(y_error.mean()),
    }

    if len(np.unique(y_error)) == 2:
        result["AUROC"] = float(roc_auc_score(y_error, risk_score))
        result["AUPRC"] = float(average_precision_score(y_error, risk_score))
    else:
        result["AUROC"] = None
        result["AUPRC"] = None

    # Select exactly the highest-risk 20%, using stable sorting for ties.
    n_flag = max(1, int(np.ceil(0.20 * len(risk_score))))
    order = np.argsort(-risk_score, kind="stable")
    flagged = np.zeros(len(risk_score), dtype=bool)
    flagged[order[:n_flag]] = True

    result["flagged_fraction"] = float(flagged.mean())
    result["flagged_error_precision"] = float(
        precision_score(y_error, flagged, zero_division=0)
    )
    result["flagged_error_recall"] = float(
        recall_score(y_error, flagged, zero_division=0)
    )
    result["flagged_error_f1"] = float(
        f1_score(y_error, flagged, zero_division=0)
    )
    result["error_rate_among_flagged"] = float(y_error[flagged].mean())
    result["error_rate_among_unflagged"] = (
        float(y_error[~flagged].mean()) if (~flagged).any() else None
    )

    return result, flagged


def build_scores(b0, b4):
    y = b0["y"]
    p0, p4 = b0["probs"], b4["probs"]

    pred0 = p0.argmax(axis=1)
    pred4 = p4.argmax(axis=1)

    confidence0 = p0.max(axis=1)
    confidence4 = p4.max(axis=1)

    error0 = (pred0 != y).astype(np.int32)
    error4 = (pred4 != y).astype(np.int32)
    disagreement = (pred0 != pred4).astype(float)

    return {
        "y": y,
        "pred0": pred0,
        "pred4": pred4,
        "confidence0": confidence0,
        "confidence4": confidence4,
        "error0": error0,
        "error4": error4,
        "disagreement": disagreement,
        "risk0": 1.0 - confidence0,
        "risk4": 1.0 - confidence4,
    }


def tune_weight(y_error, uncertainty, disagreement):
    """Select alpha on validation AUROC; test data is never used for tuning."""
    if len(np.unique(y_error)) < 2:
        raise ValueError("Validation error labels contain only one class.")

    candidates = []
    for alpha in np.linspace(0.0, 1.0, 101):
        score = alpha * uncertainty + (1.0 - alpha) * disagreement
        auc = roc_auc_score(y_error, score)
        auprc = average_precision_score(y_error, score)
        candidates.append((float(auc), float(auprc), float(alpha)))

    # AUROC primary; AUPRC breaks ties.
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best_auc, best_auprc, best_alpha = candidates[0]

    return {
        "alpha": best_alpha,
        "validation_AUROC": best_auc,
        "validation_AUPRC": best_auprc,
    }


def evaluate_split(scores, weights):
    results = {}
    flags = {}

    for model_name, error_key, risk_key in (
        ("B0", "error0", "risk0"),
        ("B4", "error4", "risk4"),
    ):
        y_error = scores[error_key]
        uncertainty = scores[risk_key]
        disagreement = scores["disagreement"]

        combined = (
            weights[model_name]["alpha"] * uncertainty
            + (1.0 - weights[model_name]["alpha"]) * disagreement
        )

        methods = {
            "confidence_only": uncertainty,
            "disagreement_only": disagreement,
            "combined": combined,
        }

        results[model_name] = {}
        for method_name, risk in methods.items():
            metrics, flagged = error_metrics(y_error, risk)
            results[model_name][method_name] = metrics
            flags[f"{model_name}_{method_name}"] = flagged

    return results, flags


def make_prediction_table(ids, scores, flags):
    y = scores["y"]
    pred0, pred4 = scores["pred0"], scores["pred4"]

    table = pd.DataFrame({
        "sample_id": ids,
        "true_label_id": y,
        "true_label": [CLASS_NAMES[int(i)] for i in y],
        "b0_pred_id": pred0,
        "b0_pred_label": [CLASS_NAMES[int(i)] for i in pred0],
        "b0_confidence": scores["confidence0"],
        "b0_correct": pred0 == y,
        "b4_pred_id": pred4,
        "b4_pred_label": [CLASS_NAMES[int(i)] for i in pred4],
        "b4_confidence": scores["confidence4"],
        "b4_correct": pred4 == y,
        "models_disagree": scores["disagreement"].astype(bool),
    })

    for name, values in flags.items():
        table[f"flagged_{name}"] = values

    return table


def main():
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Creating matched validation and test loaders...")

    b0_loaders = create_paper2_loaders(
        BATCH_SIZE_B0,
        make_eval_transform(224),
    )

    b4_loaders = create_paper2_loaders(
        BATCH_SIZE_B4,
        make_eval_transform(380),
    )

    b0_model = load_model("efficientnet_b0", B0_CHECKPOINT)
    b4_model = load_model("efficientnet_b4", B4_CHECKPOINT)

    collected = {}
    for split_name in ("val", "test"):
        b0 = collect_predictions(
            b0_model, b0_loaders[split_name], "EfficientNet-B0", split_name
        )
        b4 = collect_predictions(
            b4_model, b4_loaders[split_name], "EfficientNet-B4", split_name
        )
        verify_pairing(b0, b4, split_name)
        collected[split_name] = (
            b0,
            b4,
            build_scores(b0, b4),
        )

    val_scores = collected["val"][2]

    weights = {}
    for model_name, error_key, risk_key in (
        ("B0", "error0", "risk0"),
        ("B4", "error4", "risk4"),
    ):
        weights[model_name] = tune_weight(
            val_scores[error_key],
            val_scores[risk_key],
            val_scores["disagreement"],
        )

    # Evaluate the selected validation weights on test exactly once.
    test_ids = collected["test"][0]["ids"]
    test_scores = collected["test"][2]
    test_results, test_flags = evaluate_split(test_scores, weights)

    predictions = make_prediction_table(
        test_ids,
        test_scores,
        test_flags,
    )
    predictions.to_csv(OUTPUT_DIR / "test_predictions_model_specific_preprocessing.csv", index=False)

    report = {
        "validation_tuning": weights,
        "test_metrics": test_results,
        "evaluation_notes": [
            "Combination weights selected using validation AUROC.",
            "Test set used only for final evaluation.",
            "The current split is row-stratified; patient/lesion independence is not established.",
            "B0 evaluation uses 224x224 preprocessing and B4 evaluation uses 380x380 preprocessing, both with A.Normalize() and ToTensorV2().",
            "Sample IDs combine source and filename; they do not establish patient-level independence.",
        ],
    }

    with (OUTPUT_DIR / "error_detection_metrics_model_specific_preprocessing.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\nSelected validation weights:")
    print(json.dumps(weights, indent=2))
    print("\nFinal test metrics:")
    print(json.dumps(test_results, indent=2))
    print("\nSaved:", OUTPUT_DIR / "test_predictions_model_specific_preprocessing.csv")
    print("Saved:", OUTPUT_DIR / "error_detection_metrics_model_specific_preprocessing.json")


if __name__ == "__main__":
    main()