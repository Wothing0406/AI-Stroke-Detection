# AI Stroke Detection via Voice Analysis (AI Phân tích Giọng nói Phát hiện Đột quỵ)

## 📖 Giới thiệu (Overview)
Dự án này là một hệ thống hỗ trợ chẩn đoán sớm nguy cơ đột quỵ dựa trên phân tích đặc trưng giọng nói. Hệ thống sử dụng trí tuệ nhân tạo (AI) để phân tích các biến đổi nhỏ trong giọng nói (như độ rung, ngắt quãng, bất thường trong tần số) mà tai thường khó phát hiện.

**Luồng hoạt động (Workflow):**
1. **Người dùng** ghi âm giọng nói qua Web App.
2. **Hệ thống** gửi âm thanh về Server.
3. **AI Engine** xử lý tín hiệu -> Trích xuất đặc trưng (MFCC, Jitter, Shimmer).
4. **Mô hình ML** (SVM/Random Forest) phân loại nguy cơ.
5. **Kết quả** được trả về và hiển thị trực quan cho người dùng.

---

## 🏗️ Kiến trúc & Công nghệ (Tech Stack)

### 1. Frontend (Giao diện người dùng)
- **Framework**: React (Vite)
- **Styling**: TailwindCSS (Giao diện Y tế/Hiện đại)
- **Tính năng chính**:
  - Ghi âm trực tiếp trên trình duyệt.
  - Hiển thị sóng âm (Waveform).
  - Báo cáo kết quả phân tích.

### 2. Backend (Xử lý trung tâm)
- **Framework**: Python FastAPI (Hiệu năng cao, dễ tích hợp AI).
- **Xử lý âm thanh**: `librosa`, `soundfile` (Chuẩn hóa về 16kHz, khử nhiễu).
- **Machine Learning**: `scikit-learn` (Mô hình SVM/Random Forest).

---

## 📂 Cấu trúc Dự án (Directory Structure)

```
/
├── backend/                # Server & AI Logic
│   ├── main.py             # API Endpoint (FastAPI)
│   ├── audio_processing.py # Xử lý tín hiệu (Noise reduction, Trim)
│   ├── feature_extraction.py # Trích xuất MFCC, Jitter, Shimmer
│   ├── ml_model.py         # Huấn luyện và dự đoán (SVM/RF)
│   ├── model.pkl           # File mô hình đã huấn luyện
│   └── requirements.txt    # Danh sách thư viện Python
│
├── frontend/               # Giao diện React
│   ├── src/
│   │   ├── components/     # AudioRecorder, ResultCard...
│   │   └── App.jsx         # Màn hình chính
│   └── package.json
│
└── README.md               # Tài liệu hướng dẫn này
```

---

## 🚀 Hướng dẫn Cài đặt (Setup Guide)

### 1. Chuẩn bị (Prerequisites)
- Python 3.9+
- Node.js 16+

### 2. Cài đặt Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
# Server sẽ chạy tại: http://localhost:8000
```

### 3. Cài đặt Frontend
```bash
cd frontend
npm install
npm run dev
# Web App sẽ chạy tại: http://localhost:5173
```

---

## ⚠️ Lưu ý về Dữ liệu (Data Note)
Hiện tại dự án sử dụng **Dữ liệu Giả lập (Synthetic/Dummy Data)** để huấn luyện mô hình demo. Mục đích là để kiểm chứng luồng hoạt động của hệ thống. Để sử dụng trong y tế thực tế, cần huấn luyện lại mô hình với bộ dữ liệu bệnh án lâm sàng chuẩn.
