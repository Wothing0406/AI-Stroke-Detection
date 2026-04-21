from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Flowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib import colors
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime
import numpy as np
import uuid
import audio_processing

# --- 0. Font Setup ---
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Using DejaVuSans for UTF-8 Vietnamese character support
    font_path = os.path.join(base_dir, "font", "DejaVuSans.ttf")
    
    if not os.path.exists(font_path):
         # Fallback search
         font_path = os.path.join(base_dir, "fonts", "DejaVuSans.ttf")
    
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
        pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', font_path)) # Using same for simplicity or bold if exists
        FONT_NAME = 'DejaVuSans'
        FONT_BOLD = 'DejaVuSans-Bold'
        print(f"DEBUG: Successfully Registered {FONT_NAME} from {font_path}")
    else:
        print(f"Warning: Could not find DejaVuSans.ttf at {font_path}")
        FONT_NAME = 'Helvetica'
        FONT_BOLD = 'Helvetica-Bold'
except Exception as e:
    print(f"Warning: Could not setup fonts ({e}). Using Helvetica.")
    FONT_NAME = 'Helvetica'
    FONT_BOLD = 'Helvetica-Bold'

# --- 1. Custom Flowables ---

class RiskIndicator(Flowable):
    """
    Draws the visual risk dots (circles) in a Platypus story.
    """
    def __init__(self, risk_level="NORMAL"):
        Flowable.__init__(self)
        self.risk_level = risk_level
        self.width = 16 * cm
        self.height = 1.2 * cm

    def draw(self):
        canvas = self.canv
        circle_x = 0.5 * cm
        y_center = 0.6 * cm
        radius = 0.35 * cm
        gap = 1.1 * cm

        # Label and details based on risk
        risk_color = colors.green
        risk_text_vn = "MỨC ĐỘ SAI LỆCH: THẤP"
        fill_count = 1
        
        if self.risk_level == "HIGH":
            risk_color = colors.red
            risk_text_vn = "MỨC ĐỘ SAI LỆCH: CAO"
            fill_count = 4
        elif self.risk_level == "MEDIUM":
            risk_color = colors.orange
            risk_text_vn = "MỨC ĐỘ SAI LỆCH: TRUNG BÌNH"
            fill_count = 2

        # Draw 4 circles
        for i in range(4):
            canvas.setLineWidth(1)
            canvas.setStrokeColor(colors.black)
            if i < fill_count:
                canvas.setFillColor(risk_color)
                canvas.circle(circle_x + i * gap, y_center, radius, stroke=1, fill=1)
            else:
                canvas.setFillColor(colors.white)
                canvas.circle(circle_x + i * gap, y_center, radius, stroke=1, fill=0)

        # Draw labels
        canvas.setFont(FONT_BOLD, 14)
        canvas.setFillColor(risk_color)
        canvas.drawString(circle_x + 4.5 * cm, y_center - 0.1 * cm, risk_text_vn)
        
        canvas.setFont(FONT_NAME, 9)
        canvas.setFillColor(colors.black)
        canvas.drawString(circle_x + 4.5 * cm, y_center - 0.5 * cm, f"(Technical Risk Level: {self.risk_level})")

# --- 2. Chart Creation Helpers (Returning BytesIO) ---

def create_waveform_plot(audio_path):
    try:
        y, sr = audio_processing.load_audio(audio_path)
        if y is None: return None
        if len(y) > 10 * sr: y = y[:10*sr]
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(np.linspace(0, len(y)/sr, len(y)), y, color='#0088cc', linewidth=0.5)
        ax.set_axis_off()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=150)
        buf.seek(0)
        plt.close(fig)
        return buf
    except:
        return None

def create_radar_chart(details):
    labels = []
    values = []
    key_metrics = ["jitter_local", "shimmer_local", "hnr", "speech_rate", "mean_f0"]
    
    for key in key_metrics:
        if key in details:
            item = details[key]
            deviation = item.get('deviation_level', 0.0)
            norm_val = item.get('norm_val', min(deviation / 3.0, 1.0))
            labels.append(key.upper())
            values.append(norm_val)
    
    if not values or len(values) < 3: return None
    values = np.concatenate((values, [values[0]]))
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels([])
    ax.set_title("Deviation Profile", size=9, y=1.1)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def create_zscore_chart(details):
    labels = []
    z_scores = []
    colors_list = []
    
    for k, v in details.items():
        if k in ["jitter_local", "shimmer_local", "hnr", "speech_rate", "mean_f0", "loudness_db"]:
            z = v.get('z_score', 0)
            labels.append(k)
            z_scores.append(z)
            if abs(z) > 2: colors_list.append('#d32f2f')
            elif abs(z) > 1: colors_list.append('#f57c00')
            else: colors_list.append('#388e3c')

    if not z_scores: return None
    fig, ax = plt.subplots(figsize=(4, 3))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, z_scores, align='center', color=colors_list, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title('Feature Z-Scores', fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlim([-2.5, 2.5])
    ax.set_xlabel('Z-Score (Deviation)', fontsize=7)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

# --- 3. Main Report Generation ---

def generate_medical_report(patient_info, result, audio_path, output_path, metrics=None):
    if metrics is None: metrics = {}
    
    # Doc Template Setup
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    
    styles = getSampleStyleSheet()
    # Update default styles to use Unicode font
    for style_name in styles.byName:
        styles[style_name].fontName = FONT_NAME
        if hasattr(styles[style_name], 'boldFontName'):
             styles[style_name].boldFontName = FONT_BOLD

    # Custom Styles
    styles.add(ParagraphStyle(name='CenterTitle', fontName=FONT_BOLD, fontSize=18, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name='CenterSub', fontName=FONT_NAME, fontSize=10, alignment=TA_CENTER, spaceAfter=20))
    styles.add(ParagraphStyle(name='SectionHeader', fontName=FONT_BOLD, fontSize=12, textColor=colors.HexColor('#1a5a96'), spaceBefore=15, spaceAfter=10))
    styles.add(ParagraphStyle(name='MetricLabel', fontName=FONT_BOLD, fontSize=10))
    styles.add(ParagraphStyle(name='NormalSmall', fontName=FONT_NAME, fontSize=9))
    styles.add(ParagraphStyle(name='IDHeader', fontName=FONT_NAME, fontSize=8, alignment=TA_RIGHT))
    styles.add(ParagraphStyle(name='AlertText', fontName=FONT_NAME, fontSize=8, textColor=colors.red))
    styles.add(ParagraphStyle(name='FooterText', fontName=FONT_NAME, fontSize=7, textColor=colors.grey))

    story = []

    # --- Header Information ---
    meta = result.get('metadata', {})
    analysis_id = meta.get('analysis_id', result.get('session_id', str(uuid.uuid4())[:8]))
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    story.append(Paragraph(f"ID: {analysis_id}", styles['IDHeader']))
    story.append(Paragraph(f"Ngày: {date_str}", styles['IDHeader']))
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph("AI SPEECH ANALYTICS", styles['CenterTitle']))
    story.append(Paragraph("(Báo Cáo Kết Quả Sàng Lọc Từ Hệ Thống AI)", styles['CenterSub']))

    # --- Section I: Patient & Quality ---
    story.append(Paragraph("I. THÔNG TIN & CHẤT LƯỢNG MẪU (PATIENT & QUALITY)", styles['SectionHeader']))
    
    name = patient_info.get('name', 'N/A')
    age = patient_info.get('age', 'N/A')
    gender = patient_info.get('gender', 'N/A')
    notes = patient_info.get('health_notes', 'N/A')
    
    patient_table_data = [
        [Paragraph(f"<b>Họ tên:</b> {name}", styles['Normal']), Paragraph(f"<b>Tuổi:</b> {age}", styles['Normal']), Paragraph(f"<b>Giới tính:</b> {gender}", styles['Normal'])]
    ]
    pt_table = Table(patient_table_data, colWidths=[6.5*cm, 4*cm, 4*cm])
    pt_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'LEFT'), ('LEFTPADDING', (0,0), (-1,-1), 0)]))
    story.append(pt_table)
    
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"<b>Ghi chú lâm sàng:</b> {notes}", styles['Normal']))
    
    snr = metrics.get('snr', 0)
    vad = metrics.get('vad_ratio', 0)
    clip = metrics.get('clipping_ratio', 0)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"SNR: {snr:.2f}dB | VAD Ratio: {vad:.2f} | Clipping: {clip:.2f}", styles['NormalSmall']))

    # --- Section II: Screening Summary ---
    story.append(Paragraph("II. TỔNG HỢP SÀNG LỌC (SCREENING SUMMARY)", styles['SectionHeader']))
    
    sai = result.get('sai_score', 0)
    risk = result.get('final_risk_level', 'UNKNOWN')
    conf = result.get('confidence_score', 0)
    
    story.append(Paragraph(f"Chỉ số Sai lệch Âm thanh (SAI): <b>{sai}/100</b>", styles['Normal']))
    story.append(Spacer(1, 0.4*cm))
    
    # Custom Risk Indicator Circle Plot
    story.append(RiskIndicator(risk))
    story.append(Spacer(1, 0.5*cm))
    
    story.append(Paragraph(f"Độ tin cậy (Confidence Score): {conf*100:.1f}%", styles['Normal']))
    
    # AI Insights
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("<b>KẾT LUẬN CHI TIẾT & LỜI KHUYÊN (AI INSIGHTS)</b>", styles['Normal']))
    
    observations = result.get('observations', [])
    for obs in observations:
        story.append(Paragraph(f"• {obs}", styles['NormalSmall']))
    
    advice = result.get('advice', [])
    if advice:
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("<b>• Khuyến nghị hành động:</b>", styles['NormalSmall']))
        for adv in advice:
            story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;- {adv}", styles['NormalSmall']))
            
    explanation = result.get('explanation_text', result.get('explanation', ''))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"<i>Hệ thống phát âm bị ảnh hưởng: {explanation}</i>", styles['NormalSmall']))

    # --- Section III: Waveform ---
    story.append(Paragraph("III. BIỂU ĐỒ SÓNG ÂM (WAVEFORM)", styles['SectionHeader']))
    wave_buf = create_waveform_plot(audio_path)
    if wave_buf:
        img = Image(wave_buf, width=16*cm, height=3.5*cm)
        # Wrap image in a border table
        img_table = Table([[img]], colWidths=[16.5*cm])
        img_table.setStyle(TableStyle([('BOX', (0,0), (-1,-1), 0.5, colors.black), ('ALIGN', (0,0), (-1,-1), 'CENTER')]))
        story.append(img_table)

    # --- Section IV: Clinical Features ---
    story.append(Paragraph("IV. CHI TIẾT 54 CHỈ SỐ LÂM SÀNG (54 CLINICAL BIOMARKERS)", styles['SectionHeader']))
    story.append(Paragraph("<i>Bảng số liệu chi tiết trích xuất từ giọng nói giúp bác sĩ đánh giá cơ sở bệnh lý.</i>", styles['NormalSmall']))
    story.append(Spacer(1, 0.4*cm))
    
    details = result.get('details', result.get('detailed_metrics', {}))
    if not isinstance(details, dict): details = {}
    
    table_data = [["Chỉ số (Feature)", "Giá trị", "Tham chiếu", "Z-Score", "Trạng thái"]]
    status_map = {
        "NORMAL": "Bình thường", 
        "NEAR_BOUNDARY": "Ranh giới", 
        "DEVIATED": "Lệch nhẹ", 
        "SIGNIFICANTLY_DEVIATED": "Lệch nhiều"
    }

    for k, v in details.items():
        val = v.get('value', 0)
        z = v.get('z_score', 0)
        st_raw = v.get('status', 'NORMAL')
        st = status_map.get(st_raw, st_raw)
        ref = v.get('ref_display', 'Typical')
        label = v.get('label', k)
        
        # Color coding for status
        st_para = Paragraph(st, styles['NormalSmall'])
        if st_raw in ["DEVIATED", "SIGNIFICANTLY_DEVIATED"]:
            st_para = Paragraph(f"<font color='red'>{st}</font>", styles['NormalSmall'])
            
        table_data.append([
            Paragraph(label, styles['NormalSmall']), 
            f"{val:.4f}", 
            ref, 
            f"{z:.2f}", 
            st_para
        ])
        
    # Col widths: Name, Value, Ref, Z, Status
    ft_table = Table(table_data, colWidths=[5*cm, 3*cm, 3.5*cm, 2.5*cm, 3.5*cm], repeatRows=1)
    ft_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), FONT_BOLD),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('TOPPADDING', (0,0), (-1,0), 8),
    ]))
    story.append(ft_table)

    # --- Section V: Visual Analytics ---
    story.append(Paragraph("V. PHÂN TÍCH TRỰC QUAN (VISUAL ANALYTICS)", styles['SectionHeader']))
    
    radar_buf = create_radar_chart(details)
    zchart_buf = create_zscore_chart(details)
    
    visuals_data = []
    row = []
    if radar_buf:
        row.append(Image(radar_buf, width=8*cm, height=7*cm))
    if zchart_buf:
        row.append(Image(zchart_buf, width=8*cm, height=6*cm))
    
    if row:
        visuals_data.append(row)
        v_table = Table(visuals_data, colWidths=[8.5*cm, 8.5*cm])
        v_table.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(v_table)

    # --- Footer ---
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("<hr/>", styles['Normal'])) # Horizontal line hack
    story.append(Paragraph("<b>THÔNG TIN KỸ THUẬT (METADATA):</b>", styles['NormalSmall']))
    
    m_ver = meta.get('model_version', result.get('model_version', '1.0'))
    f_set = meta.get('feature_set_version', result.get('feature_set_version', 'N/A'))
    norm = meta.get('normalization_method', result.get('normalization_method', 'N/A'))
    
    story.append(Paragraph(f"Model: {m_ver} | Features: {f_set} | Normalization: {norm}", styles['FooterText']))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("SẢN PHẨM: AI SPEECH ANALYTICS (PHIÊN BẢN NGHIÊN CỨU 2026)", styles['MetricLabel']))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("<i>Báo cáo này chỉ mang tính chất sàng lọc và nghiên cứu, không phải chẩn đoán y khoa. Vui lòng tham khảo ý kiến chuyên gia y tế để được đánh giá chính thức.</i>", styles['AlertText']))

    # Build PDF
    doc.build(story)
    return output_path
