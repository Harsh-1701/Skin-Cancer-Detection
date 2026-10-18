# Skin Lesion Classification System

A deep learning project for classifying skin lesion images into nine categories using EfficientNet. The project includes an image prediction interface, model evaluation, Grad-CAM visualizations, and PDF report generation.

The aim was to build a system that brings model prediction and evaluation together in one application, rather than keeping them as separate scripts.

## About the Project

Skin lesions can have similar visual appearances, which makes their classification a challenging image recognition task. This project explores how convolutional neural networks can be used to classify skin lesion images and how visual explanations can help users understand a model's predictions.

Two EfficientNet architectures, B0 and B4, were implemented and evaluated. EfficientNet-B0 is used for image prediction in the application, while EfficientNet-B4 is included for comparison.

The system is intended for academic and experimental use. Its predictions should not be treated as a medical diagnosis.

## Main Features

- **Skin lesion classification:** Predicts one of nine supported classes from an uploaded image.
- **EfficientNet models:** Includes training and evaluation workflows for EfficientNet-B0 and EfficientNet-B4.
- **Grad-CAM visualization:** Generates a heatmap to highlight image regions that influence the model's prediction.
- **Model evaluation:** Displays evaluation metrics, confusion matrices, and training curves.
- **PDF reports:** Generates a report containing the prediction, confidence score, input image, and Grad-CAM visualization.
- **Streamlit interface:** Provides a single interface for image analysis, model evaluation, and project information.

## Dataset

The project uses a combined skin lesion image dataset based on HAM10000 and ISIC 2019 data, along with an `unknown` category for images outside the supported lesion classes.

The prepared dataset contains **36,346 images across nine classes**.

| Class | Description |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |
| `scc` | Squamous cell carcinoma |
| `unknown` | Images assigned to the unknown category |

The dataset was divided into training, validation, and test sets using a 70:15:15 split with random seed 42.

| Split | Number of images |
|---|---:|
| Training | 25,442 |
| Validation | 5,452 |
| Testing | 5,452 |
| **Total** | **36,346** |

The dataset and trained model checkpoints are not included in this repository. They must be prepared or obtained separately before running the complete system.

## Models and Results

Both architectures were evaluated on the same test split. The following results are from the project's verified evaluation runs.

| Metric | EfficientNet-B0 | EfficientNet-B4 |
|---|---:|---:|
| Test accuracy | 80.96% | 79.48% |
| Weighted precision | 81.08% | 79.19% |
| Weighted recall | 80.96% | 79.48% |
| Weighted F1-score | 79.76% | 79.06% |
| Macro precision | 79.87% | 65.73% |
| Macro recall | 68.18% | 69.95% |
| Macro F1-score | 72.20% | 67.13% |

EfficientNet-B0 achieved the higher overall test accuracy and weighted F1-score in these experiments, so it was selected for application inference. EfficientNet-B4 remains part of the project for comparison.

Overall accuracy alone does not describe performance equally across all classes. For example, the measured melanoma recall was 46.34% for B0 and 55.56% for B4. This difference is one reason to examine per-class results and confusion matrices alongside aggregate metrics.

These results apply to the evaluated dataset and split. They do not establish clinical reliability or performance on images from unrelated sources.

## Technology Stack

- **Python** for the project implementation
- **PyTorch** for model training and inference
- **timm** for EfficientNet model architectures
- **Albumentations** for image preprocessing
- **OpenCV and Pillow** for image handling
- **pytorch-grad-cam** for Grad-CAM visualization
- **Streamlit** for the web interface
- **ReportLab** for PDF report generation
- **scikit-learn, NumPy and pandas** for evaluation and data processing
- **Matplotlib** for plots and visualizations

## Project Structure

```text
Skin-Cancer-Detection/
├── src/
│   ├── app/
│   │   ├── app.py
│   │   └── pdf_report.py
│   ├── evaluation/
│   │   ├── evaluate_b0_improved.py
│   │   ├── evaluate_b0_verified.py
│   │   ├── evaluate_b4.py
│   │   ├── evaluate_b4_verified.py
│   │   ├── gradcam.py
│   │   └── save_classification_reports.py
│   ├── models/
│   │   └── efficientnet.py
│   ├── preprocessing/
│   │   ├── create_combined_dataset.py
│   │   ├── create_dataloader.py
│   │   ├── skin_dataset.py
│   │   └── transforms.py
│   ├── training/
│   │   ├── loss.py
│   │   ├── train.py
│   │   ├── train_config.py
│   │   └── trainer.py
│   └── utils/
│       └── config.py
├── outputs/
├── requirements.txt
├── PROJECT_PROGRESS.md
└── README.md
```

## Evaluation Outputs

The evaluation scripts generate metrics and visualizations used to compare the models. Depending on the scripts that have been run, outputs include:

- Confusion matrices for EfficientNet-B0 and EfficientNet-B4
- Training accuracy and loss curves
- Classification reports
- Model comparison results in CSV and JSON formats

These files are generated locally and may not be present in a fresh clone.

## Limitations

- The models are trained and evaluated on a particular dataset and split. Performance may differ on images from other sources.
- Class distributions are imbalanced, which can affect results for less frequent classes.
- Some classes have lower recall than others, including melanoma and squamous cell carcinoma.
- Grad-CAM provides a visual indication of image regions influencing a prediction; it does not prove that the model is using medically meaningful features.
- The system has not been validated for clinical use and must not be used to make treatment decisions.

## Future Improvements

Potential improvements include evaluating on an independent external dataset, addressing class imbalance, improving recall for underrepresented classes, and testing the system across different image acquisition conditions.

## Contributors

- **Harsh Singh** — 1MV23EC050
- **Ayush Raj** — 1MV23EC029
- **Swetangi Ray** — 1MV23EC115

**Project guide:** Dr. Amit Gangopadhyay  
Professor, Electronics and Communication Engineering  
SMVIT

## License

No license has been specified for this repository yet. Unless a license is added, the code should not be assumed to be available for unrestricted reuse.