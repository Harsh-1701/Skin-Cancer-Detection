import cv2
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

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"]{
        font-size:22px !important;
     font-weight:600;
    }

    div[data-testid="stMetricLabel"]{
        font-size:17px;
    }

    .block-container{
        padding-top:1rem;
        padding-bottom:1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
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
    padding-top:0.5rem;
    padding-bottom:0rem;
}

.card{
    background:white;
    padding:18px;
    border-radius:15px;
    box-shadow:0px 0px 15px rgba(0,0,0,.08);
}

div.stButton > button{
    width:100%;
    border-radius:12px;
    height:55px;
    font-size:18px;
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

    st.markdown("### Developed By")

    st.markdown("""
    - Harsh Singh
    - Ayush Raj
    - Swetangi Ray
    """)

    st.markdown("### Guide")

    st.write("Dr. Amit Gangopadhyay")

    st.divider()

# -------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------

if selected == "Dashboard":

    st.markdown(
        """
        <h1 style='font-size:46px; margin-bottom:0px;'>
        🩺 Skin Cancer Detection System
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
    """
    <div style="font-size:22px;
    color:#bdbdbd;
    margin-top:-10px;
    margin-bottom:20px;">

    Deep Learning Based Skin Lesion Classification using EfficientNet-B0

    </div>
    """,
    unsafe_allow_html=True,
    )

    st.divider()

    c1,c2,c3,c4 = st.columns(4)

    with c1:

        st.markdown(
        """
        ### 🎯 Accuracy

        ## 87.95%
        """
        )

    with c2:

        st.markdown(
            """
            ### 📊 Classes

            ## 7
            """
        )

    with c3:

        st.markdown(
            """
            ### 🗂 Dataset

            ## HAM10000
            """
        )

    with c4:

        st.markdown(
            """
            ### 🧠 Model

            ## EfficientNet-B0
            """
        )

    st.divider()

    st.info(
    """

    ✨ Features

    • Automatic lesion classification
    
    • Grad-CAM visualization for explainable AI
    
    • Confidence score estimation
    
    • Downloadable PDF diagnostic report

    """
    )

    st.markdown("---")

    st.markdown(
    """
    <div style="
    font-size:18px;
    text-align:center;
    color:gray;
    margin-top:20px;">

    Academic Major Project |
    Electronics and Communication Engineering |
    2026

    </div>
    """,
    unsafe_allow_html=True,
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
                caption="Input Skin Lesion Image",
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

                all_probabilities = (
                    probabilities.squeeze()
                    .detach()
                    .cpu()
                    .numpy()
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
            confidence_percent = score * 100

            if confidence_percent >= 90:
                confidence_color = "🟢"

            elif confidence_percent >= 70:
                confidence_color = "🟡"

            else:
                confidence_color = "🔴"

             # Reject only when the model is very uncertain
            if score < 0.45:

                st.warning(
                    """
            ⚠️ Low confidence prediction.

            The uploaded image may not be a skin lesion or may be outside the model's training data.

            Please upload a clear dermoscopic or clinical skin lesion image.
            """
                )

                st.stop()

            with middle:

                st.image(
                    heatmap,
                    caption="Grad-CAM Attention Map",
                    use_container_width=True,
                )

            with right:

                st.markdown("## 📋 Prediction Report")

                st.markdown("---")

                st.markdown("### 🦠 Disease")

                st.markdown(
                    f"## {DISEASE_NAMES[disease]}"
                )

                if CLASS_INFO[disease] == "Benign":

                    st.success("🟢 Benign Lesion")

                else:

                    st.error("🔴 Malignant Lesion")

                st.markdown("### 🎯 Confidence")

                st.progress(score)

                confidence_percent = score * 100

                if confidence_percent >= 90:
                    icon = "🟢"

                elif confidence_percent >= 70:
                    icon = "🟡"

                else:
                    icon = "🔴"

                st.metric(
                    "",
                    f"{icon} {confidence_percent:.2f}%"
                )

                if score >= 0.70:

                    st.success(
                        "🟢 High confidence prediction."
                    )

                elif score >= 0.50:

                    st.warning(
                        "🟡 Moderate confidence prediction. Clinical verification is recommended."
                    )

                else:

                    st.error(
                        "🔴 Low confidence prediction. Clinical verification is recommended."
                    )

                st.markdown("### 📊 Class Probabilities")

                probability_dict = {}

                for i, prob in enumerate(all_probabilities):

                    label = INDEX_TO_LABEL[i]

                    probability_dict[
                        DISEASE_NAMES[label]
                    ] = prob

                probability_dict = dict(

                    sorted(

                        probability_dict.items(),

                        key=lambda x: x[1],

                        reverse=True,

                    )

                )

                for disease_name, prob in probability_dict.items():

                    st.write(
                        f"**{disease_name}**"
                    )

                    st.progress(
                        float(prob)
                    )

                    st.caption(
                        f"{prob*100:.2f}%"
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

                        label="📄 Download Diagnostic Report (PDF)",

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
        st.markdown(
        """
        <h3 style="font-size:28px;">
        🎯 Accuracy
        </h3>
        """,
        unsafe_allow_html=True,
        )
        st.markdown(
        """
        <h2 style="font-size:34px;margin-top:0;">
        87.95%
        </h2>
        """,
        unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
        """
        <h3 style="font-size:28px;">
        📊 Classes
        </h3>
        """,
        unsafe_allow_html=True,
        )
        st.markdown(
        """
        <h2 style="font-size:34px;margin-top:0;">
        7
        </h2>
        """,
        unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
        """
        <h3 style="font-size:28px;">
        📁 Dataset
        </h3>
        """,
        unsafe_allow_html=True,
        )
        st.markdown(
        """
        <h2 style="font-size:34px;margin-top:0;">
        HAM10000
        </h2>
        """,
        unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
        """
        <h3 style="font-size:28px;">
        🧠 Model
        </h3>
        """,
        unsafe_allow_html=True,
        )
        st.markdown(
        """
        <h2 style="font-size:34px;margin-top:0;">
        EfficientNet-B0
        </h2>
        """,
        unsafe_allow_html=True,
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

    st.title("About Project")

    st.markdown("""
### AI Skin Cancer Detection System

This project classifies dermoscopic skin lesion images using EfficientNet-B0 with Transfer Learning.

---

### 📊 Project Information

- **Dataset:** HAM10000
- **Model:** EfficientNet-B0
- **Classes:** 7
- **Validation Accuracy:** 87.95%

---

### 🛠 Technologies

- Python
- PyTorch
- Streamlit
- OpenCV
- Grad-CAM
- ReportLab

---

### 👨‍💻 Developed By

- Harsh Singh (1MV23EC050)
- Ayush Raj (1MV23EC029)
- Swetangi Ray (1MV23EC115)

---

### 👨‍🏫 Guide

- Dr. Amit Gangopadhyay

Department of Electronics & Communication Engineering
""")