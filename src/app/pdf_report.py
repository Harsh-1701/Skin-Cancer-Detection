from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib.colors import darkblue
from reportlab.lib.styles import ParagraphStyle

import os

def generate_pdf(

    pdf_path,

    uploaded_image,

    gradcam_image,

    disease,

    lesion_type,

    confidence,

    description,

):

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(

        "Title",

        parent=styles["Heading1"],

        alignment=TA_CENTER,

        textColor=darkblue,

        spaceAfter=20,

    )

    doc = SimpleDocTemplate(pdf_path)

    story = []

    story.append(
        Paragraph(
            "AI Skin Cancer Detection Report",
            title_style,
        )
    )

    story.append(Spacer(1, 0.3 * inch))

    if os.path.exists(uploaded_image):

        story.append(
            Image(
                uploaded_image,
                width=2.6 * inch,
                height=2.6 * inch,
            )
        )

    story.append(Spacer(1, 0.2 * inch))

    if os.path.exists(gradcam_image):

        story.append(
            Image(
                gradcam_image,
                width=2.6 * inch,
                height=2.6 * inch,
            )
        )

    story.append(Spacer(1, 0.3 * inch))

    story.append(
        Paragraph(
            f"<b>Disease:</b> {disease}",
            styles["Normal"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Type:</b> {lesion_type}",
            styles["Normal"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Confidence:</b> {confidence:.2f}%",
            styles["Normal"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Description:</b> {description}",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 0.3 * inch))

    story.append(
        Paragraph(
            "Model : EfficientNet-B0",
            styles["Normal"],
        )
    )

    story.append(
        Paragraph(
            "Dataset : HAM10000",
            styles["Normal"],
        )
    )

    story.append(
        Paragraph(
            "Developed By : Harsh, Ayush & Swetangi",
            styles["Normal"],
        )
    )

    doc.build(story)