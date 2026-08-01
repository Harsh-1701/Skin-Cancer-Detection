import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from src.models.efficientnet import build_model
from src.utils.config import (
    CLASS_INFO,
    INDEX_TO_LABEL,
    DISEASE_NAMES,
    DISEASE_DESCRIPTION,
)

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="AI Skin Cancer Detection",
    page_icon="🩺",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "outputs/models/best_model.pth",
        map_location=DEVICE,
    )

    model = build_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model


model = load_model()

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.title("📋 Project Information")

    st.markdown("### Model")
    st.write("EfficientNet-B0")

    st.markdown("### Dataset")
    st.write("HAM10000")

    st.markdown("### Classes")
    st.write("7 Skin Diseases")

    st.markdown("### Validation Accuracy")
    st.success("87.95 %")

    st.markdown("### Made with ❤️ by...")
    st.write("Harsh, Ayush & Swetangi")


# -----------------------------
# Header
# -----------------------------
st.title("🩺 Skin Cancer Detection System")

st.caption(
    "Deep Learning based Skin Lesion Classification using EfficientNet-B0"
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Dermoscopic Image",
    type=["jpg", "jpeg", "png"],
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns([1, 1])

    with left:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True,
        )

        predict = st.button(
            "🔍 Predict",
            use_container_width=True,
        )

    with right:

        st.subheader("Prediction Report")

        if predict:

            image_np = np.array(image)

            image_tensor = transform(image_np)

            image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(DEVICE)

            with torch.no_grad():

                output = model(image_tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )

                confidence, prediction = torch.max(
                    probabilities,
                    dim=1,
                )

            disease = INDEX_TO_LABEL[prediction.item()]
            score = confidence.item()

            st.metric(
                "Disease",
                DISEASE_NAMES[disease],
            )

            if CLASS_INFO[disease] == "Benign":

                st.success("🟢 Benign")

            else:

                st.error("🔴 Malignant")

            st.write("### Confidence")

            st.progress(score)

            st.metric(
                "Confidence Score",
                f"{score*100:.2f}%",
            )

            st.info(
                DISEASE_DESCRIPTION[disease]
            )

            report = f"""
AI Skin Cancer Detection Report

-----------------------------------

Disease:
{DISEASE_NAMES[disease]}

Type:
{CLASS_INFO[disease]}

Confidence:
{score*100:.2f} %

Model:
EfficientNet-B0

Dataset:
HAM10000
"""

            st.download_button(
                "📄 Download Report",
                data=report,
                file_name="prediction_report.txt",
            )