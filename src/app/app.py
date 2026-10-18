import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.app.pdf_report import generate_pdf
from src.evaluation.gradcam import GradCAMGenerator
from src.utils.config import INDEX_TO_LABEL, DISEASE_NAMES, DISEASE_DESCRIPTION

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model_b0_final.pth"
PLOT_DIR = PROJECT_ROOT / "outputs" / "plots"
NUM_CLASSES = 9

st.set_page_config(
    page_title="Skin Lesion Classification System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .main-heading {
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 700;
        letter-spacing: -1.2px;
        line-height: 1.15;
        margin-bottom: .5rem;
    }
    .subheading {
        font-size: 1.05rem;
        opacity: .78;
        margin-bottom: 1.5rem;
    }
    .section-heading {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: .5rem;
    }
    .soft-panel {
        padding: 1.15rem 1.25rem;
        border: 1px solid var(--border-color, rgba(128,128,128,.25));
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .muted-text {
        opacity: .72;
        font-size: .9rem;
    }
    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }
    div[role="radiogroup"] {
        gap: 1.2rem;
    }
    div[role="radiogroup"] label {
        font-weight: 550;
    }
    .stButton button, .stDownloadButton button {
        border-radius: 8px;
        min-height: 2.7rem;
        font-weight: 600;
    }
    hr {
        margin-top: 1.3rem;
        margin-bottom: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=NUM_CLASSES,
    )

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_gradcam(_model):
    return GradCAMGenerator(_model)


transform = A.Compose([
    A.Resize(height=380, width=380),
    A.Normalize(),
    ToTensorV2(),
])

try:
    model = load_model()
    gradcam = load_gradcam(model)
except Exception as exc:
    st.error("Could not load the EfficientNet-B0 model.")
    st.code(str(exc))
    st.stop()


def heading(title, subtitle=None):
    st.markdown(f'<div class="main-heading">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="subheading">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def section(title):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )


def create_pdf(result):
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        original_path = temp / "uploaded_image.png"
        heatmap_path = temp / "gradcam_image.png"
        pdf_path = temp / "skin_lesion_report.pdf"

        result["image"].save(original_path)

        if result["heatmap"] is not None:
            Image.fromarray(result["heatmap"]).save(heatmap_path)

        generate_pdf(
            pdf_path=str(pdf_path),
            uploaded_image=str(original_path),
            gradcam_image=str(heatmap_path) if result["heatmap"] is not None else None,
            disease=result["name"],
            lesion_type=result["key"],
            confidence=result["confidence"],
            description=result["description"],
        )
        return pdf_path.read_bytes()


heading(
    "Skin Lesion Classification System",
    "Image-based classification using deep learning",
)

page = st.radio(
    "Navigation",
    ["Analyze Image", "Model Evaluation", "About"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

with st.sidebar:
    st.markdown("### Project Team: 13")
    st.caption("This project has been developed by:")

    st.markdown("""
    **Harsh Singh**  
    *(1MV23EC050)*

    **Ayush Raj**  
    *(1MV23EC029)*

    **Swetangi Ray**  
    *(1MV23EC115)*
    """)

    st.divider()
    st.markdown("### Under the Guidance of")

    st.markdown("""
    **Dr. Amit Gangopadhyay**  
    Professor, ECE  
    SMVIT
    """)

# ---------------------------------------------------------
# ANALYZE IMAGE: MAIN LANDING PAGE
# ---------------------------------------------------------
if page == "Analyze Image":
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        section("Analyze a skin image")
        st.write(
            "Upload an image to obtain the model's predicted category, "
            "confidence score, and visual explanation."
        )

        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG and PNG.",
        )

        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
                image_np = np.asarray(image)

                st.image(
                    image,
                    caption=f"Uploaded image · {image.width} × {image.height} pixels",
                    use_container_width=True,
                )

                if st.button(
                    "Analyze Image",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Analyzing image..."):
                        model_input = transform(image=image_np)["image"]
                        model_input = model_input.unsqueeze(0).to(DEVICE)

                        with torch.no_grad():
                            logits = model(model_input)
                            probabilities = torch.softmax(logits, dim=1)[0]

                        score, prediction = torch.max(probabilities, dim=0)
                        key = INDEX_TO_LABEL[int(prediction.item())]

                        try:
                            heatmap = gradcam.generate(model_input, image_np)
                        except Exception:
                            heatmap = None

                        st.session_state["result"] = {
                            "key": key,
                            "name": DISEASE_NAMES[key],
                            "confidence": float(score.item()) * 100,
                            "description": DISEASE_DESCRIPTION[key],
                            "probabilities": probabilities.detach().cpu().numpy(),
                            "image": image.copy(),
                            "heatmap": heatmap,
                        }

            except Exception as exc:
                st.error("Unable to read this image.")
                st.code(str(exc))

    with right:
        section("How it works")
        st.markdown("""
        <div class="soft-panel">
            <b>01 &nbsp; Upload</b><br>
            <span class="muted-text">Select a JPG or PNG image.</span>
        </div>
        <div class="soft-panel">
            <b>02 &nbsp; Classify</b><br>
            <span class="muted-text">EfficientNet-B0 estimates the most likely supported category.</span>
        </div>
        <div class="soft-panel">
            <b>03 &nbsp; Review</b><br>
            <span class="muted-text">Inspect the model scores and Grad-CAM visualization, then export a PDF.</span>
        </div>
        """, unsafe_allow_html=True)

    result = st.session_state.get("result")

    if result:
        st.divider()
        section("Analysis result")

        result_left, result_right = st.columns([1, 1], gap="large")

        with result_left:
            st.markdown("**Predicted category**")
            st.subheader(result["name"])
            st.metric("Model confidence score", f'{result["confidence"]:.2f}%')
            st.write(result["description"])
            st.caption(
                "The score represents the model's output, not a confirmed diagnosis."
            )

        with result_right:
            if result["heatmap"] is not None:
                st.image(
                    result["heatmap"],
                    caption="Grad-CAM visualization",
                    use_container_width=True,
                )
            else:
                st.info("Grad-CAM visualization could not be generated for this image.")

        with st.expander("View scores for all categories"):
            scores = result["probabilities"]
            for i in range(NUM_CLASSES):
                key = INDEX_TO_LABEL[i]
                st.write(
                    f'**{DISEASE_NAMES[key]}** · {scores[i] * 100:.2f}%'
                )
                st.progress(float(scores[i]))

        pdf_bytes = create_pdf(result)
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name="skin_lesion_report.pdf",
            mime="application/pdf",
            type="primary",
        )


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------
elif page == "Model Evaluation":
    heading(
        "Model Evaluation",
        "Comparison of EfficientNet-B0 and EfficientNet-B4 on the held-out test set.",
    )

    st.markdown("### Performance comparison")

    comparison = [
        {
            "Metric": "Test accuracy",
            "EfficientNet-B0": "80.96%",
            "EfficientNet-B4": "79.48%",
        },
        {
            "Metric": "Weighted F1-score",
            "EfficientNet-B0": "0.7976",
            "EfficientNet-B4": "0.7906",
        },
        {
            "Metric": "Macro F1-score",
            "EfficientNet-B0": "0.7220",
            "EfficientNet-B4": "0.6713",
        },
        {
            "Metric": "Macro recall",
            "EfficientNet-B0": "0.6818",
            "EfficientNet-B4": "0.6995",
        },
        {
            "Metric": "Melanoma recall",
            "EfficientNet-B0": "46.34%",
            "EfficientNet-B4": "55.56%",
        },
        {
            "Metric": "SCC recall",
            "EfficientNet-B0": "30.85%",
            "EfficientNet-B4": "28.72%",
        },
    ]

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        "**Selected prediction model: EfficientNet-B0.** "
        "It achieved higher overall test accuracy and higher weighted and macro "
        "F1-scores in this evaluation."
    )
    st.caption(
        "B4 had higher melanoma recall in this test split. Results depend on the "
        "dataset and evaluation setup; neither model is clinically validated."
    )

    st.divider()
    section("Confusion matrices")

    col_b0, col_b4 = st.columns(2, gap="large")

    with col_b0:
        st.markdown("**EfficientNet-B0**")
        path = PLOT_DIR / "confusion_matrix_b0_verified.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info("B0 confusion matrix not found.")

    with col_b4:
        st.markdown("**EfficientNet-B4**")
        path = PLOT_DIR / "confusion_matrix_b4_verified.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
        else:
            st.info("B4 confusion matrix not found.")

    st.divider()
    section("Training curves")

    curve_b0, curve_b4 = st.columns(2, gap="large")

    with curve_b0:
        st.markdown("**EfficientNet-B0**")
        for filename, caption in [
            ("accuracy_curve.png", "Accuracy"),
            ("loss_curve.png", "Loss"),
        ]:
            path = PLOT_DIR / filename
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)

    with curve_b4:
        st.markdown("**EfficientNet-B4**")
        for filename, caption in [
            ("accuracy_curve_b4.png", "Accuracy"),
            ("loss_curve_b4.png", "Loss"),
        ]:
            path = PLOT_DIR / filename
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)


# ---------------------------------------------------------
# ABOUT
# ---------------------------------------------------------
elif page == "About":
    heading(
        "About the Project",
        "Deep-learning-based classification of skin-lesion images.",
    )

    section("Project overview")
    st.write(
        "This project investigates multi-class skin-lesion classification using "
        "EfficientNet architectures. EfficientNet-B0 is used for the current "
        "image-analysis workflow because it achieved stronger overall results "
        "in the project's test evaluation."
    )

    st.divider()
    section("Project details")

    details_left, details_right = st.columns(2, gap="large")

    with details_left:
        st.markdown("**Model**")
        st.write("EfficientNet-B0")
        st.markdown("**Task**")
        st.write("Nine-category image classification")
        st.markdown("**Interface**")
        st.write("Streamlit")

    with details_right:
        st.markdown("**Datasets**")
        st.write("HAM10000, ISIC2019, and an additional unknown-image category")
        st.markdown("**Libraries**")
        st.write("Python, PyTorch, timm, Albumentations, Grad-CAM, ReportLab")

    st.divider()
    st.caption(
        "Academic research project. Model predictions are experimental outputs "
        "and should not replace assessment by a qualified healthcare professional."
    )
