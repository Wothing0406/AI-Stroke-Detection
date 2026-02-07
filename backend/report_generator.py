from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import io
import os
from datetime import datetime
import librosa
import numpy as np

# Register Vietnamese Font
try:
    # Try different paths to find the font
    font_path = os.path.join("backend", "fonts", "arial.ttf")
    if not os.path.exists(font_path):
        font_path = os.path.join("fonts", "arial.ttf")
    if not os.path.exists(font_path):
        # Fallback to absolute path if needed, or C:\Windows\Fonts
        font_path = r"C:\Windows\Fonts\arial.ttf"
        
    pdfmetrics.registerFont(TTFont('Arial', font_path))
    FONT_NAME = 'Arial'
    FONT_BOLD = 'Arial' # Arial usually includes bold, but we'll use same for now or register bold if needed
except Exception as e:
    print(f"Warning: Could not load Arial font ({e}). Using Helvetica.")
    FONT_NAME = 'Helvetica'
    FONT_BOLD = 'Helvetica-Bold'

def generate_medical_report(patient_info, result, audio_path, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    
    # --- 1. Header (Chuyên nghiệp) ---
    # Logo place holder (optional)
    
    c.setFont(FONT_BOLD, 22)
    c.drawCentredString(width/2, height - 2*cm, "BÁO CÁO PHÂN TÍCH GIỌNG NÓI Y KHOA")
    c.setFont(FONT_NAME, 10)
    c.drawCentredString(width/2, height - 2.5*cm, "(HỆ THỐNG SÀNG LỌC ĐỘT QUỴ QUA GIỌNG NÓI - AI POWERED)")
    
    # Horizontal Line
    c.setLineWidth(1)
    c.line(2*cm, height - 3*cm, width - 2*cm, height - 3*cm)
    
    c.setFont(FONT_NAME, 10)
    c.drawString(2*cm, height - 3.5*cm, f"Ngày tạo: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    c.drawString(2*cm, height - 4.0*cm, f"Mã hồ sơ: {os.path.basename(output_path).replace('.pdf', '')}")

    # --- 2. Patient Information (Thông tin Bệnh nhân) ---
    y_pos = height - 5.5*cm
    c.setFont(FONT_BOLD, 14)
    c.setFillColorRGB(0, 0, 0.5) # Dark Blue
    c.drawString(2*cm, y_pos, "1. THÔNG TIN BỆNH NHÂN")
    c.setFillColorRGB(0, 0, 0)
    
    c.setFont(FONT_NAME, 12)
    y_pos -= 0.8*cm
    c.drawString(2.5*cm, y_pos, f"Họ và tên: {patient_info.get('name', 'N/A')}")
    y_pos -= 0.7*cm
    c.drawString(2.5*cm, y_pos, f"Ngày sinh: {patient_info.get('dob', 'N/A')}")
    y_pos -= 0.7*cm
    c.drawString(2.5*cm, y_pos, f"CCCD/CMND: {patient_info.get('cccd', 'N/A')}")
    
    # --- 3. AI Analysis Result (Kết quả Phân tích) ---
    y_pos -= 1.5*cm
    c.setFont(FONT_BOLD, 14)
    c.setFillColorRGB(0, 0, 0.5)
    c.drawString(2*cm, y_pos, "2. KẾT QUẢ PHÂN TÍCH AI")
    c.setFillColorRGB(0, 0, 0)
    
    # Risk Assessment
    y_pos -= 1.0*cm
    risk_label = result['risk_assessment']
    
    if risk_label == 'High Risk':
        risk_text = "NGUY CƠ CAO - CÓ DẤU HIỆU BẤT THƯỜNG"
        details_text = "Hệ thống phát hiện các đặc trưng âm thanh không ổn định (Jitter/Shimmer cao), thường gặp ở người có vấn đề về thần kinh cơ hoặc từng bị đột quỵ."
        color = (0.8, 0.1, 0.1) # Red
    else:
        risk_text = "NGUY CƠ THẤP - GIỌNG NÓI BÌNH THƯỜNG"
        details_text = "Các chỉ số âm thanh nằm trong giới hạn bình thường. Không phát hiện dấu hiệu rối loạn vận ngôn rõ ràng."
        color = (0.1, 0.6, 0.1) # Green
        
    c.setFont(FONT_BOLD, 12)
    c.drawString(2.5*cm, y_pos, "Chẩn đoán Sơ bộ:")
    
    c.setFont(FONT_BOLD, 14)
    c.setFillColorRGB(*color)
    c.drawString(6.5*cm, y_pos, risk_text)
    c.setFillColorRGB(0, 0, 0) # Reset
    
    y_pos -= 0.8*cm
    c.setFont(FONT_NAME, 12)
    c.drawString(2.5*cm, y_pos, f"Độ tin cậy (Confidence): {result['confidence_score']*100:.1f}%")
    
    y_pos -= 0.8*cm
    c.drawString(2.5*cm, y_pos, "Chi tiết lâm sàng:")
    c.setFont(FONT_NAME, 11)
    # Wrap text if needed, but for now just print
    c.drawString(2.5*cm, y_pos - 0.5*cm, details_text)

    # --- 4. Waveform Visualization (Biểu đồ Sóng âm) ---
    y_pos -= 3*cm
    c.setFont(FONT_BOLD, 14)
    c.setFillColorRGB(0, 0, 0.5)
    c.drawString(2*cm, y_pos, "3. BIỂU ĐỒ SÓNG ÂM (VOICE WAVEFORM)")
    c.setFillColorRGB(0, 0, 0)
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        plt.figure(figsize=(8, 3))
        plt.plot(np.linspace(0, len(y)/sr, len(y)), y, alpha=0.7, color='teal')
        plt.title("Phân tích Biên độ theo Thời gian", fontsize=10)
        plt.xlabel("Thời gian (giây)", fontsize=8)
        plt.ylabel("Biên độ", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150)
        img_buf.seek(0)
        plt.close()
        
        img = ImageReader(img_buf)
        c.drawImage(img, 2*cm, y_pos - 8*cm, width=17*cm, height=6.5*cm)
    except Exception as e:
        c.setFont(FONT_NAME, 10)
        c.drawString(2.5*cm, y_pos - 1*cm, f"Không thể tạo biểu đồ: {str(e)}")

    # --- 5. Footer (Chân trang) ---
    c.setLineWidth(0.5)
    c.line(2*cm, 2.5*cm, width - 2*cm, 2.5*cm)
    
    c.setFont(FONT_NAME, 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    disclaimer = "LƯU Ý QUAN TRỌNG: Báo cáo này được tạo tự động bởi AI nhằm mục đích sàng lọc sớm."
    disclaimer2 = "Kết quả KHÔNG thay thế cho chẩn đoán của bác sĩ chuyên khoa. Vui lòng đến cơ sở y tế nếu có nghi ngờ."
    c.drawCentredString(width/2, 2*cm, disclaimer)
    c.drawCentredString(width/2, 1.6*cm, disclaimer2)
    
    c.setFont(FONT_NAME, 8)
    c.drawCentredString(width/2, 1.0*cm, "Developed by Nguyen Duy Quang | Hotline: 0795277227")

    c.save()
    return output_path
