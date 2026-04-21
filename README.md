# 🧠 HỆ THỐNG TẦM SOÁT SỚM NGUY CƠ ĐỘT QUỴ - AI STROKE DETECTION
## ỨNG DỤNG KIẾN TRÚC HYBRID AI VÀ PHÂN TÍCH CHỈ SỐ SINH HỌC GIỌNG NÓI LÂM SÀNG

[![Docker](https://img.shields.io/badge/Docker-enabled-blue.svg)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Scientific--v2.0-orange.svg)]()

> **Tác giả**: Nguyễn Duy Quang  
> **Lĩnh vực**: Phần mềm hệ thống (Trí tuệ nhân tạo Y sinh (Biomedical AI) - Công nghệ hỗ trợ sức khỏe)  
> **Phiên bản**: 2.0 (Scientific Version)
> **Trạng thái**: Research & Development
> **Cấp độ**: Chuyên nghiệp / Khoa học (Scientific Excellence Layer)

---

## 📑 MỤC LỤC
1. [**Vấn đề nghiên cứu và Bối cảnh Y khoa**](#1-vấn-đề-nghiên-cứu-và-bối-cảnh-y-khoa)
2. [**Thiết kế hệ thống và Phương pháp luận**](#2-thiết-kế-hệ-thống-và-phương-pháp-luận)
3. [**Thực hiện: Chế tạo và Kiểm tra**](#3-thực-hiện-chế-tạo-và-kiểm-tra)
    - a. Đặc tả kỹ thuật và Stack Công nghệ (Modern Micro-Acoustic Stack)
    - b. **Hướng dẫn Cài đặt & Triển khai Docker**
    - c. **Tự Treo Host với Cloudflare Tunnel**
    - d. Quy trình xử lý tín hiệu (Medical Signal Processing Pipeline)
    - e. Hệ thống Báo cáo Y khoa Chuyên nghiệp (Automated A4 Clinical PDF)
4. [**Kết luận và Hướng phát triển chiến lược**](#4-kết-luận-và-hướng-phát-triển-chiến-lược)

---

## 1. VẤN ĐỀ NGHIÊN CỨU VÀ BỐI CẢNH Y KHOA

### a. Thách thức trong tầm soát đột quỵ tại cộng đồng
Đột quỵ (Stroke) là một tình trạng cấp cứu y khoa nghiêm trọng xảy ra khi dòng máu đến não bị gián đoạn. Tại Việt Nam, các nghiên cứu chỉ ra rằng phần lớn bệnh nhân nhập viện ngoài "Thời gian vàng" do sự nhận thức hạn chế về các triệu chứng vận động lời nói và thiếu hụt công cụ sàng lọc tại chỗ.

### b. Cơ sở sinh lý học: Mối liên hệ giữa Thần kinh và Giọng nói (Bio-Acoustic Link)
Sự phát âm là một quá trình tinh vi đòi hỏi sự phối hợp đồng bộ giữa hệ hô hấp, thanh quản và các bộ phận cấu âm. Khi xảy ra đột quỵ, sự biến thiên cực nhỏ trong mili-giây mà tai người không thể phân biệt nhưng các thuật toán xử lý tín hiệu số (DSP) có thể định lượng được thông qua các chỉ số như Jitter, Shimmer và MFCC.

---

## 2. THIẾT KẾ HỆ THỐNG VÀ PHƯƠNG PHÁP LUẬN

### a. Quy trình trích xuất Đặc trưng Lâm sàng (Digital Biomarkers Pipeline)
Kiến trúc hệ thống được xây dựng để xử lý dữ liệu theo tầng, đảm bảo tính toàn vẹn của tín hiệu từ khi ghi âm đến khi trích xuất 54 biomarkers chuyên sâu.

### b. Chi tiết 54 Chỉ số Sinh học Giọng nói (Unified Feature Vector)
Hệ thống trích xuất một tập hợp tham số âm học đa chiều:
- **Nhóm 1: Pitch & Frequency Stability** (Jitter, F0 Var...)
- **Nhóm 2: Amplitude Stability** (Shimmer, APQ...)
- **Nhóm 3: Harmonic Structure** (HNR, CPP...)
- **Nhóm 4: Spectral Shape** (Spectral Centroid, MFCC 1-13...)
- **Nhóm 5: Temporal Dynamics** (Speech Rate, Pause Ratio, VOT...)
- **Nhóm 6: Voice Quality Indicators** (Formant F1-F3, Stability...)

### c. Kiến trúc Hybrid AI Consensus Engine (SAI Architecture)
Hệ thống sử dụng mô hình "Hội đồng AI" (Consensus Engine) kết hợp giữa:
1. **Isolation Forest**: Học không giám sát để phát hiện các mẫu dị thường (Anomaly Detection).
2. **Random Forest**: Học có giám sát để phân loại bệnh lý dựa trên 54 chiều đặc trưng.

---

## 3. THỰC HIỆN: CHẾ TẠO VÀ KIỂM TRA

### a. Đặc tả kỹ thuật và Stack Công nghệ
- **Backend**: FastAPI (Python 3.9), Librosa, Scikit-learn, ReportLab.
- **Frontend**: React 18, Vite, Tailwind CSS, Recharts, Framer Motion.
- **Infrastructure**: **Docker & Docker Compose** (Containerization).

### b. Hướng dẫn Cài đặt & Triển khai Docker
Hệ thống đã được đóng gói toàn bộ vào Docker để dễ dàng triển khai trên mọi môi trường:

## 🚀 Hướng dẫn Cài đặt (Docker)

Để chạy dự án này trên máy tính của bạn, hãy đảm bảo bạn đã cài đặt **Docker** và **Docker Compose**.

1. **Khởi chạy hệ thống**:
   ```bash
   docker compose up --build
   ```
2. **Truy cập**:
   - Giao diện người dùng: [http://localhost](http://localhost)
   - API Backend: [http://localhost:8000](http://localhost:8000)

### c. Tự Treo Host với Cloudflare Tunnel
Hệ thống tích hợp sẵn Cloudflared để tạo link công khai HTTPS miễn phí mà không cần mở port modem:
1. Sau khi chạy Docker, kiểm tra log của container `tunnel`:
   ```bash
   docker compose logs -f tunnel
   ```
2. Tìm link có dạng: `https://xxxx.trycloudflare.com`. Đây là link công khai của bạn.

### d. Quy trình xử lý tín hiệu (Medical Signal Processing Pipeline)
Mỗi tệp âm thanh trải qua quy trình 5 giai đoạn: Normalization -> Multi-stage Denoising -> Intelligent VAD -> Parallel Feature Extraction -> Consensus Scoring.

### e. Hệ thống Báo cáo Y khoa Chuyên nghiệp (Automated A4 Clinical PDF)
Hệ thống tự động hóa việc tạo báo cáo chuyên nghiệp mẫu A4 bao gồm phân tích Waveform, biểu đồ Radar Biomarkers, biểu đồ Z-Score và các nhận xét AI tự động.

---

## 4. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN CHIẾN LƯỢC

Dự án giúp tăng tỷ lệ phát hiện sớm trong "Thời gian vàng" và giảm tải cho hệ thống y tế thông qua sàng lọc từ xa. Lộ trình tương lai bao gồm tích hợp **Multi-modal AI** (kết hợp giọng nói và hình ảnh khuôn mặt) để đạt độ chính xác lâm sàng >98%.

---
*Dự án tâm huyết được thực hiện bởi Nguyễn Duy Quang | Kế thừa và Phát triển trên nền tảng Công nghệ Xử lý Tín hiệu Y sinh 2026*
