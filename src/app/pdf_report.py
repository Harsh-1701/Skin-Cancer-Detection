import os
import html
import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    KeepTogether,
)


NAVY = colors.HexColor("#142D4E")
BLUE = colors.HexColor("#247A9B")
TEXT = colors.HexColor("#26364A")
MUTED = colors.HexColor("#65758B")
LIGHT = colors.HexColor("#F3F7FA")
BORDER = colors.HexColor("#DCE5ED")
WARNING_BG = colors.HexColor("#FFF5DF")
WARNING_BORDER = colors.HexColor("#E8C66D")
WHITE = colors.white


def _safe_text(value):
    """Escape user- or model-derived text before using it in a PDF paragraph."""
    return html.escape(str(value if value is not None else ""))


def _fit_image(path, max_width, max_height):
    """Fit an image inside a box without changing its aspect ratio."""
    if not path or not os.path.isfile(path):
        return None

    img = Image(path)
    scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale
    return img


def generate_pdf(
    pdf_path,
    uploaded_image,
    gradcam_image,
    disease,
    lesion_type,
    confidence,
    description,
):
    page_width, page_height = A4
    left_margin = 18 * mm
    right_margin = 18 * mm
    usable_width = page_width - left_margin - right_margin

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=right_margin,
        leftMargin=left_margin,
        topMargin=17 * mm,
        bottomMargin=19 * mm,
        title="Skin Lesion Classification Research Report",
        author="Team 13",
    )

    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=base["Normal"],
        fontSize=9,
        leading=13,
        textColor=MUTED,
        spaceAfter=3,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=NAVY,
        spaceBefore=11,
        spaceAfter=7,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontSize=9,
        leading=14,
        textColor=TEXT,
        spaceAfter=3,
    )

    label_style = ParagraphStyle(
        "Label",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=NAVY,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=body_style,
        fontSize=8,
        leading=11,
        textColor=MUTED,
    )

    warning_style = ParagraphStyle(
        "Warning",
        parent=body_style,
        fontSize=8.5,
        leading=12,
        textColor=TEXT,
    )

    story = []
    generated_at = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")

    # Header
    story.append(Paragraph("Skin Lesion Analysis", title_style))
    story.append(Paragraph("CNN assisted classification research report", subtitle_style))
    story.append(Paragraph(f"Generated: {_safe_text(generated_at)}", small_style))
    story.append(Spacer(1, 5 * mm))

    header_rule = Table([[""]], colWidths=[usable_width], rowHeights=[1.2 * mm])
    header_rule.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(header_rule)
    story.append(Spacer(1, 3 * mm))

    # Model output
    story.append(Paragraph("Model Output", section_style))

    summary = [
        [Paragraph("Predicted category", label_style),
         Paragraph(_safe_text(disease), body_style)],
        [Paragraph("Dataset label", label_style),
         Paragraph(_safe_text(lesion_type), body_style)],
        [Paragraph("Model confidence score", label_style),
         Paragraph(f"{float(confidence):.2f}%", body_style)],
    ]

    summary_table = Table(
        summary,
        colWidths=[usable_width * 0.38, usable_width * 0.62],
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)

    story.append(Paragraph("Category Information", section_style))
    description_text = _safe_text(description)
    desc_table = Table(
        [[Paragraph(description_text, body_style)]],
        colWidths=[usable_width],
    )
    desc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("LINEBEFORE", (0, 0), (0, -1), 3, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(desc_table)

    # Image section, with aspect ratios preserved
    original = _fit_image(
        uploaded_image,
        max_width=(usable_width / 2) - 12 * mm,
        max_height=65 * mm,
    )
    heatmap = _fit_image(
        gradcam_image,
        max_width=(usable_width / 2) - 12 * mm,
        max_height=65 * mm,
    )

    if original is not None or heatmap is not None:
        story.append(Paragraph("Image Review", section_style))

        cell_width = usable_width / 2
        image_cells = [
            original if original is not None else Paragraph("Original image unavailable.", small_style),
            heatmap if heatmap is not None else Paragraph(
                "Grad-CAM visualization unavailable.", small_style
            ),
        ]
        captions = [
            Paragraph("<b>Uploaded image</b>", small_style),
            Paragraph("<b>Grad-CAM visualization</b>", small_style),
        ]

        image_table = Table(
            [image_cells, captions],
            colWidths=[cell_width, cell_width],
        )
        image_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, 0), 0.5, BORDER),
            ("INNERGRID", (0, 0), (-1, 0), 0.5, BORDER),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("BACKGROUND", (0, 1), (-1, 1), LIGHT),
        ]))
        story.append(image_table)
        story.append(Paragraph(
            "Grad-CAM highlights image regions associated with the model's output. "
            "It does not establish why a prediction is correct and is not clinical evidence.",
            small_style,
        ))

    # Project metadata
    story.append(Paragraph("Project Information", section_style))

    project_rows = [
        [Paragraph("Model architecture", label_style),
         Paragraph("EfficientNet-B0", body_style)],
        [Paragraph("Task", label_style),
         Paragraph("Nine-category skin-lesion image classification", body_style)],
        [Paragraph("Dataset scope", label_style),
         Paragraph("Combined HAM10000, ISIC2019, and unknown-image category", body_style)],
        [Paragraph("Interface", label_style),
         Paragraph("Streamlit", body_style)],
    ]

    project_table = Table(
        project_rows,
        colWidths=[usable_width * 0.38, usable_width * 0.62],
    )
    project_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(project_table)
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph(
        "For academic demonstration only. This system is not validated for clinical use. "
        "A model score must not be interpreted as a confirmed diagnosis.",
        small_style,
    ))

    def draw_page(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(
            left_margin,
            13 * mm,
            page_width - right_margin,
            13 * mm,
        )
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(left_margin, 8 * mm, "Harsh | Ayush | Swetangi")
        canvas.drawRightString(
            page_width - right_margin,
            8 * mm,
            f"Page {document.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
