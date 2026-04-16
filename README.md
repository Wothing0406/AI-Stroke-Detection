# 🧠 AI Stroke Detection System
## Ứng dụng Kiến trúc Hybrid AI và Phân tích Digital Biomarkers Giọng nói

[![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Scientific--v2.0-orange.svg)]()

Hệ thống tầm soát sớm nguy cơ đột quỵ thông qua phân tích 54 chỉ số sinh học giọng nói (Digital Biomarkers). Sử dụng mô hình Hybrid AI để đạt được độ chính xác khoa học và có khả năng tự triển khai (Self-hosting) thông qua Docker.

---

## 📑 Mục lục
1. [Giới thiệu & Cơ sở Khoa học](#giới-thiệu--cơ-sở-khoa-học)
2. [Tính năng Chính](#tính-năng-chính)
3. [Công nghệ Sử dụng](#công-nghệ-sử-dụng)
4. [Hướng dẫn Cài đặt (Docker)](#hướng-dẫn-cài-đặt)
5. [Tự Treo Host với Cloudflare Tunnel](#tự-treo-host-với-cloudflare-tunnel)
6. [Cấu trúc Dự án](#cấu-trúc-dự-án)
7. [Tác giả](#tác-giả)

---

## 🔬 Giới thiệu & Cơ sở Khoa học

Dự án này tập trung vào việc nhận diện các biến đổi vi mô trong giọng nói (micro-vocal changes) mà tai người thường không nhận ra, nhưng lại là dấu hiệu sớm của các tổn thương thần kinh liên quan đến đột quỵ.

- **54 Biomarkers**: Trích xuất các chỉ số về Pitch Stability (Jitter), Amplitude Dynamics (Shimmer), Harmonic Structure (HNR/CPP), và Spectral Shape (MFCC).
- **Hybrid AI Engine**: Sự kết hợp giữa Isolation Forest (phát hiện dị thường) và Random Forest (phân loại bệnh lý).
- **Age-Based Calibration**: Tự động hiệu chuẩn ngưỡng cảnh báo theo độ tuổi để giảm thiểu tỷ lệ dương tính giả ở người cao tuổi.

---

## ✨ Tính năng Chính

- 🎙️ **Thu âm chuẩn hóa**: Quy trình 3 bước (Nguyên âm, Đếm số, Đọc câu) để thu thập dữ liệu âm học đa chiều.
- 📊 **Phân tích thời gian thực**: Trực quan hóa phổ âm thanh và chỉ số rủi ro (SAI Score) ngay khi thu âm.
- 📄 **Báo cáo Y khoa PDF**: Tự động xuất báo cáo chuyên nghiệp kèm biểu đồ Radar so sánh với chỉ số chuẩn.
- ☁️ **Self-hosting Ready**: Được đóng gói toàn bộ vào Docker, dễ dàng triển khai lên máy chủ cá nhân hoặc VPS.

---

## 🛠️ Công nghệ Sử dụng

### Frontend
- **React 18 + Vite**: Giao diện Dashboard hiện đại, mượt mà.
- **Tailwind CSS**: Thiết kế responsive, UI/UX chuẩn y khoa.
- **Lucide Icons & Framer Motion**: Iconography và micro-animations cao cấp.

### Backend
- **FastAPI (Python 3.9)**: API hiệu năng cao với xử lý bất đồng bộ.
- **Librosa & DSP**: Xử lý tín hiệu số và trích xuất đặc trưng âm học.
- **Scikit-learn**: Vận hành mô hình Hybrid AI.
- **ReportLab**: Tạo báo cáo PDF tự động.

---

## 🚀 Hướng dẫn Cài đặt (Docker)

Để chạy dự án này trên máy tính của bạn, hãy đảm bảo bạn đã cài đặt **Docker** và **Docker Compose**.

1. **Clone repository**:
   ```bash
   git clone <your-repo-url>
   cd dotquy
   ```

2. **Chạy hệ thống**:
   ```bash
   docker compose up --build
   ```

3. **Truy cập**:
   - Giao diện người dùng: [http://localhost](http://localhost)
   - API Backend: [http://localhost:8000](http://localhost:8000)

---

## 🌐 Tự Treo Host với Cloudflare Tunnel

Hệ thống đã tích hợp sẵn **Cloudfare Tunnel**, giúp bạn có một link công khai cho mọi người sử dụng hoàn toàn miễn phí mà không cần mở port modem.

1. Sau khi chạy `docker compose up`, hãy kiểm tra log của container `tunnel`:
   ```bash
   docker compose logs -f tunnel
   ```
2. Tìm dòng chữ có dạng: `https://your-unique-name.trycloudflare.com`.
3. Chia sẻ đường link này cho mọi người. Bất kỳ ai cũng có thể truy cập hệ thống đang chạy trên máy của bạn thông qua link này với HTTPS bảo mật.

---

## 📁 Cấu trúc Dự án

```text
.
├── backend/            # FastAPI Backend & AI Logic
├── frontend/           # React Dashboard
├── docker-compose.yml  # Orchestration
├── README.md           # Documentation
└── .gitignore          # Git exclusion rules
```

---

## 👤 Tác giả

**Nguyễn Duy Quang**  
*Lĩnh vực: Biomedical AI & Healthcare Technology*  
*Năm thực hiện: 2026*

---
> [!NOTE]
> Kết quả từ hệ thống này chỉ mang tính chất tầm soát sớm và tham khảo khoa học, không thay đổi các chẩn đoán y khoa chính thống từ bác sĩ chuyên môn.
