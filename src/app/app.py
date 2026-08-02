import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from streamlit_option_menu import option_menu

from src.app.pdf_report import generate_pdf
from src.models.efficientnet import build_model
from src.evaluation.gradcam import GradCAMGenerator
from src.utils.config import (
    CLASS_INFO,
    INDEX_TO_LABEL,
    DISEASE_NAMES,
    DISEASE_DESCRIPTION,
)

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="AI Skin Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown(
    """
<style>

.main{
    background:#f5f8fc;
}

.block-container{
    padding-top:1rem;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(0,0,0,.08);
}

.big-font{
    font-size:32px;
    font-weight:bold;
}

.small-font{
    color:gray;
}

div.stButton > button{
    width:100%;
    border-radius:12px;
    height:55px;
    font-size:20px;
}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

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

gradcam = GradCAMGenerator(model)

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]
)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

with st.sidebar:

    selected = option_menu(

        menu_title="Navigation",

        options=[
            "Dashboard",
            "Prediction",
            "Evaluation",
            "About"
        ],

        icons=[
            "house",
            "activity",
            "bar-chart",
            "info-circle"
        ],

        default_index=0,
    )

    st.divider()

    st.markdown("### Model")

    st.success("EfficientNet-B0")

    st.markdown("### Validation Accuracy")

    st.success("87.95 %")

    st.markdown("### Dataset")

    st.info("HAM10000")

    st.markdown("### Classes")

    st.info("7")

    st.markdown("### Device")

    if torch.cuda.is_available():
        st.success("GPU")
    else:
        st.warning("CPU")

    st.divider()

    st.caption("Developed By")

    st.write("Harsh")

    st.write("Ayush")

    st.write("Swetangi")

# -------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------

if selected == "Dashboard":

    st.markdown(
        "<div class='big-font'>🩺 AI Skin Cancer Detection System</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='small-font'>Deep Learning Based Skin Lesion Classification using EfficientNet-B0</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    c1,c2,c3,c4 = st.columns(4)

    with c1:

        st.metric(
            "Validation Accuracy",
            "87.95%"
        )

    with c2:

        st.metric(
            "Classes",
            "7"
        )

    with c3:

        st.metric(
            "Dataset",
            "HAM10000"
        )

    with c4:

        st.metric(
            "Architecture",
            "EfficientNet-B0"
        )

    st.divider()

    st.info(
        """
This application classifies dermoscopic skin lesion images into
seven different disease categories using a deep learning model.

Navigate to **Prediction** from the left menu to upload an image.
"""
    )

    # -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

elif selected == "Prediction":

    st.title("🔍 Skin Lesion Prediction")

    uploaded_file = st.file_uploader(
        "Upload Dermoscopic Image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(image)

        image_tensor = transform(
            image_np
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            DEVICE
        )

        left, middle, right = st.columns(3)

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

        if predict:

            with torch.enable_grad():

                output = model(
                    image_tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )

                confidence, prediction = torch.max(
                    probabilities,
                    dim=1,
                )

            heatmap = gradcam.generate(
                image_tensor,
                image_np,
            )
            from PIL import Image

            Image.fromarray(image_np).save(
                "outputs/uploaded_image.png"
            )

            Image.fromarray(heatmap).save(
                "outputs/gradcam_image.png"
            )

            disease = INDEX_TO_LABEL[
                prediction.item()
            ]

            score = confidence.item()

            with middle:

                st.image(
                    heatmap,
                    caption="Grad-CAM",
                    use_container_width=True,
                )

            with right:

                st.subheader(
                    "Prediction Report"
                )

                st.metric(
                    "Disease",
                    DISEASE_NAMES[disease],
                )

                if CLASS_INFO[disease] == "Benign":

                    st.success(
                        "🟢 Benign"
                    )

                else:

                    st.error(
                        "🔴 Malignant"
                    )

                st.progress(score)

                st.metric(
                    "Confidence",
                    f"{score*100:.2f}%"
                )

                st.info(
                    DISEASE_DESCRIPTION[
                        disease
                    ]
                )

                generate_pdf(

                    pdf_path="outputs/prediction_report.pdf",

                    uploaded_image="outputs/uploaded_image.png",

                    gradcam_image="outputs/gradcam_image.png",

                    disease=DISEASE_NAMES[disease],

                    lesion_type=CLASS_INFO[disease],

                    confidence=score * 100,

                    description=DISEASE_DESCRIPTION[disease],

                )

                with open(
                    "outputs/prediction_report.pdf",
                    "rb",
                ) as pdf:

                    st.download_button(

                        "📄 Download PDF Report",

                        data=pdf,

                        file_name="AI_Skin_Cancer_Report.pdf",

                        mime="application/pdf",

                        use_container_width=True,

                    )

    # -------------------------------------------------------
# EVALUATION
# -------------------------------------------------------

elif selected == "Evaluation":

    st.title("📊 Model Evaluation")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Accuracy",
            "87.95%"
        )

    with c2:
        st.metric(
            "Precision",
            "87.98%"
        )

    with c3:
        st.metric(
            "Recall",
            "87.95%"
        )

    with c4:
        st.metric(
            "F1 Score",
            "87.92%"
        )

    st.divider()

    st.subheader("Confusion Matrix")

    st.image("outputs/plots/diagnosis_distribution.png")

    st.divider()

    st.subheader("Training Accuracy")

    st.image(
        "outputs/plots/accuracy_curve.png",
        use_container_width=True,
    )

    st.divider()

    st.subheader("Training Loss")

    st.image(
        "outputs/plots/loss_curve.png",
        use_container_width=True,
    )


# -------------------------------------------------------
# ABOUT
# -------------------------------------------------------

elif selected == "About":

    st.title("ℹ About Project")

    st.markdown("""
### AI Skin Cancer Detection System

This project classifies dermoscopic skin lesion images using
EfficientNet-B0 with Transfer Learning.

### Dataset

HAM10000

### Model

EfficientNet-B0

### Number of Classes

7

### Validation Accuracy

87.95%

### Technologies

- Python
- PyTorch
- Streamlit
- OpenCV
- Grad-CAM
- ReportLab

### Developers

- Harsh
- Ayush
- Swetangi

### Guide

Department of Computer Engineering
""")