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
    # Professional modern color palette
    PRIMARY_COLOR = colors.HexColor("#1A365D")   # Deep clinical navy
    SECONDARY_COLOR = colors.HexColor("#2B6CB0") # Clean slate blue
    TEXT_DARK = colors.HexColor("#2D3748")       # Dark charcoal for readability
    TEXT_MUTED = colors.HexColor("#718096")      # Neutral grey for labels/metadata
    BG_LIGHT = colors.HexColor("#F8FAFC")        # Soft off-white for table rows
    BORDER_COLOR = colors.HexColor("#E2E8F0")    # Modern subtle borders

    # Document setup with elegant 0.75 inch margins
    doc = SimpleDocTemplate(
        pdf_path,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    # Define clean, modern typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        textColor=PRIMARY_COLOR,
        fontSize=24,
        leading=28,
        fontName="Helvetica-Bold",
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        textColor=TEXT_MUTED,
        fontSize=10,
        leading=14,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        textColor=PRIMARY_COLOR,
        fontSize=12,
        leading=16,
        fontName="Helvetica-Bold",
        spaceBefore=12,
        spaceAfter=6,
    )

    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        textColor=TEXT_DARK,
        fontSize=10,
        leading=14,
    )

    cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )

    desc_style = ParagraphStyle(
        "DescText",
        parent=styles["BodyText"],
        textColor=TEXT_DARK,
        fontSize=10,
        leading=15,
    )

    footer_style = ParagraphStyle(
        "FooterText",
        parent=styles["Normal"],
        textColor=TEXT_MUTED,
        fontSize=8,
        leading=12,
        alignment=TA_CENTER,
    )

    story = []

    # --- HEADER SECTION ---
    story.append(Paragraph("Skin Cancer Detection Report", title_style))
    current_time = datetime.datetime.now().strftime('%d %B %Y | %I:%M %p')
    story.append(Paragraph(f"<b>Generated on:</b> {current_time}", subtitle_style))
    story.append(Spacer(1, 0.15 * inch))

    # Clean divider line
    divider = Table([[""]], colWidths=[7 * inch])
    divider.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, -1), 1.5, PRIMARY_COLOR),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 0.2 * inch))

    # Helper function to wrap text in Paragraphs for flawless table formatting
    def wrap_text(text, style):
        return Paragraph(str(text), style)

    # --- PREDICTION SUMMARY SECTION ---
    story.append(Paragraph("Prediction Summary", heading_style))
    
    summary_data = [
        [wrap_text("Predicted Disease", cell_bold), wrap_text(disease, cell_style)],
        [wrap_text("Lesion Type", cell_bold), wrap_text(lesion_type, cell_style)],
        [wrap_text("Confidence Score", cell_bold), wrap_text(f"{confidence:.2f}%", cell_style)],
    ]

    summary_table = Table(summary_data, colWidths=[2.3 * inch, 4.7 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER_COLOR),
                ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    # --- IMAGE ANALYSIS SECTION ---
    images = []
    if os.path.exists(uploaded_image):
        images.append(Image(uploaded_image, width=2.8 * inch, height=2.4 * inch))
    if os.path.exists(gradcam_image):
        images.append(Image(gradcam_image, width=2.8 * inch, height=2.4 * inch))

    if len(images) == 2:
        story.append(Paragraph("Image Analysis", heading_style))
        
        # Wrapped side-by-side images with visual frame
        image_table_data = [
            images,
            [
                wrap_text("<b>Figure 1:</b> Uploaded Clinical Image", subtitle_style),
                wrap_text("<b>Figure 2:</b> Grad-CAM Activation Map", subtitle_style)
            ]
        ]
        image_table = Table(image_table_data, colWidths=[3.5 * inch, 3.5 * inch])
        image_table.setStyle(
            TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(image_table)
        story.append(Spacer(1, 0.2 * inch))

    # --- CLINICAL DESCRIPTION SECTION ---
    story.append(Paragraph("Clinical Description", heading_style))
    
    # Beautiful callout card layout for the text description
    desc_table_data = [[wrap_text(description, desc_style)]]
    desc_table = Table(desc_table_data, colWidths=[7.0 * inch])
    desc_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
            ("LINELEFT", (0, 0), (0, -1), 3, SECONDARY_COLOR), # Left accent border
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 14),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ])
    )
    story.append(desc_table)
    story.append(Spacer(1, 0.25 * inch))

    # --- PROJECT INFORMATION SECTION ---
    story.append(Paragraph("Project Information", heading_style))

    info = [
        [wrap_text("Model Architecture", cell_bold), wrap_text("EfficientNet-B0", cell_style)],
        [wrap_text("Dataset Used", cell_bold), wrap_text("HAM10000", cell_style)],
        [wrap_text("Department", cell_bold), wrap_text("Electronics and Communication Engineering", cell_style)],
        [wrap_text("Project Guide", cell_bold), wrap_text("Dr. Amit Gangopadhyay", cell_style)],
    ]

    info_table = Table(info, colWidths=[2.3 * inch, 4.7 * inch])
    info_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), BG_LIGHT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER_COLOR),
            ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
        ])
    )
    story.append(info_table)
    story.append(Spacer(1, 0.2 * inch))

    # --- DEVELOPERS & FOOTER ---
    story.append(Paragraph("Developed By", heading_style))
    
    # Sleek inline row style instead of bullet points to save space and look modern
    devs_text = "<b>Harsh Singh</b> &nbsp;&bull;&nbsp; <b>Ayush Raj</b> &nbsp;&bull;&nbsp; <b>Swetangi Ray</b>"
    story.append(Paragraph(devs_text, cell_style))
    story.append(Spacer(1, 0.25 * inch))

    # Disclaimer note
    disclaimer = (
        "<i>This report is generated automatically by the Skin Cancer Detection System "
        "for academic demonstration purposes and should not be considered a clinical medical diagnosis.</i>"
    )
    story.append(Paragraph(disclaimer, footer_style))

    doc.build(story)