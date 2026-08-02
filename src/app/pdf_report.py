from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib.colors import (
    darkblue,
    lightgrey,
    black,
    whitesmoke,
)

from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors

import datetime
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
        fontSize=22,
        leading=26,
        spaceAfter=12,
    )

    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        textColor=darkblue,
        spaceAfter=10,
    )

    doc = SimpleDocTemplate(pdf_path)

    story = []

    story.append(
        Paragraph(
            "Skin Cancer Detection Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            f"<b>Generated on : 02 August 2026 | 11:25 AM:</b> {datetime.datetime.now().strftime('%d %B %Y | %I:%M %p')}",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 0.18 * inch))

    story.append(
        Paragraph(
            "Prediction Summary",
            heading,
        )
    )

    data = [
        ["Disease", disease],
        ["Lesion Type", lesion_type],
        ["Confidence", f"{confidence:.2f}%"],
    ]

    table = Table(
        data,
        colWidths=[2.2 * inch, 3.8 * inch],
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), lightgrey),
                ("BACKGROUND", (1, 0), (1, -1), whitesmoke),
                ("GRID", (0, 0), (-1, -1), 1, black),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    story.append(table)

    story.append(Spacer(1, 0.18 * inch))

    story.append(
        Paragraph(
            "Image Analysis",
            heading,
        )
    )

    images = []

    if os.path.exists(uploaded_image):

        images.append(
            Image(
                uploaded_image,
                width=2.6 * inch,
                height=2.6 * inch,
            )
        )

    if os.path.exists(gradcam_image):

        images.append(
            Image(
                gradcam_image,
                width=2.6 * inch,
                height=2.6 * inch,
            )
        )

    if len(images) == 2:

        image_table = Table(
            [images],
            colWidths=[3 * inch, 3 * inch],
        )

        image_table.setStyle(

            TableStyle([

                ("ALIGN", (0,0), (-1,-1), "CENTER"),

                ("BOTTOMPADDING", (0,0), (-1,-1), 10),

            ])

        )

        story.append(image_table)

    story.append(Spacer(1,0.18*inch))

    story.append(Spacer(1, 0.18 * inch))

    story.append(
        Paragraph(
            "Clinical Description",
            heading,
        )
    )

    story.append(
        Paragraph(
            description,
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 0.18 * inch))

    story.append(
        Paragraph(
            "Project Information",
            heading,
        )
    )

    info = [

        ["Model", "EfficientNet-B0"],

        ["Dataset", "HAM10000"],

        ["Department", "Electronics and Communication Engineering"],

        ["Guide", "Dr. Amit Gangopadhyay"],

    ]

    info_table = Table(

        info,

        colWidths=[2.3*inch,3.7*inch]

    )

    info_table.setStyle(

        TableStyle([

            ("GRID",(0,0),(-1,-1),1,black),

            ("BACKGROUND",(0,0),(0,-1),lightgrey),

            ("BACKGROUND",(1,0),(1,-1),whitesmoke),

            ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),

            ("BOTTOMPADDING",(0,0),(-1,-1),8),

        ])

    )

    story.append(info_table)

    story.append(Spacer(1,0.18*inch))

    story.append(
        Paragraph(
            "<b>Developed By</b>",
            heading,
        )
    )

    story.append(
        Paragraph(
            "• Harsh Singh<br/>"
            "• Ayush Raj<br/>"
            "• Swetangi Ray",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 0.18 * inch))

    story.append(
        Paragraph(
            "<font color='grey'><i>This report is generated automatically by the Skin Cancer Detection System for academic demonstration purposes and should not be considered a medical diagnosis.</i></font>",
            styles["BodyText"],
        )
    )

    doc.build(story)