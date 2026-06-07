# Hệ thống Định danh Khuôn mặt Thời gian Thực

> **Trường Đại học Quy Nhơn – Khoa Công nghệ Thông tin – Bộ môn Trí tuệ Nhân tạo**  
> Sinh viên: Trần Thanh Tiến | MSSV: 4651050270  
> GVHD: Lê Thị Kim Nga

---

## Tổng quan

Hệ thống nhận diện và điểm danh khuôn mặt thời gian thực sử dụng:

| Thành phần | Công nghệ |
|---|---|
| Phát hiện khuôn mặt | RetinaFace (qua InsightFace) |
| Trích xuất đặc trưng | buffalo_l / MobileFaceNet (512-D embedding) |
| Chống giả mạo | CNN 5 tầng lightweight (texture + Laplacian) |
| So khớp | Cosine Similarity (threshold = 0.6) |
| Lưu trữ | SQLite (cooldown 5 phút) |
| Giao diện | OpenCV real-time display |

---

## Cấu trúc thư mục

```
face_recognition_system/
│
├── main.py                    ← File chạy chính (entry point)
├── face_recognition_system.py ← Class FaceRecognitionSystem + evaluate_condition()
├── anti_spoofing.py           ← Module chống giả mạo (CNN 5 tầng)
├── database.py                ← SQLite handler (persons + attendance)
├── utils.py                   ← Hàm tiện ích (resize, draw, FPS, cosine...)
├── requirements.txt           ← Danh sách thư viện
├── README.md                  ← Tài liệu này
│
├── attendance.db              ← Tự động tạo khi chạy lần đầu
└── ~/.insightface/
    └── models/buffalo_l/      ← Model tự động tải (~500MB)
```

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8 – 3.11 (khuyến nghị 3.10)
- Webcam hoặc camera USB
- RAM tối thiểu: 4GB (khuyến nghị 8GB)
- GPU NVIDIA (tùy chọn, tăng FPS đáng kể)

### Bước 1 – Tạo môi trường ảo

```bash
# Tạo venv
python -m venv venv

# Kích hoạt (Linux/Mac)
source venv/bin/activate

# Kích hoạt (Windows)
venv\Scripts\activate
```

### Bước 2 – Cài thư viện

```bash
pip install -r requirements.txt
```

> **Lần đầu chạy:** InsightFace tự động tải model `buffalo_l` (~500MB).  
> Cần kết nối Internet. Sau khi tải xong, hoạt động offline.

### Bước 3 (tùy chọn) – Dùng GPU

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

---

## Hướng dẫn sử dụng

### 1. Đăng ký khuôn mặt

**Từ camera (khuyến nghị):**
```bash
python main.py --mode register
```
Hệ thống sẽ hỏi mã số và họ tên, sau đó chụp 5 mẫu ảnh tự động.

**Từ file ảnh:**
```bash
python main.py --mode register_img \
    --image "anh_the.jpg" \
    --id "SV4651050270" \
    --name "Tran Thanh Tien" \
    --role student
```

### 2. Chạy nhận diện

```bash
python main.py
# hoặc
python main.py --mode run --camera 0
```

**Phím tắt trong cửa sổ camera:**

| Phím | Chức năng |
|---|---|
| `Q` / `ESC` | Thoát chương trình |
| `S` | Đăng ký khuôn mặt từ frame hiện tại |
| `A` | Xem danh sách điểm danh hôm nay (terminal) |
| `R` | Reload gallery từ database |

### 3. Kiểm tra 6 điều kiện hệ thống

```bash
python main.py --mode evaluate
```

Output mẫu:
```
=================================================================
  EVALUATE CONDITIONS - Kiểm tra 6 điều kiện hệ thống
=================================================================

✓ PASS | Condition 1: Frame Resize 640×480
       Input: 1920×1080 → Output: 640×480

✓ PASS | Condition 2: Cosine Similarity Threshold = 0.6
       Cùng người: sim=0.9842 ≥ 0.6 → OK  |  Khác người: sim=0.0312 < 0.6 → OK

✓ PASS | Condition 3: Anti-Spoofing Threshold = 0.75
       Anh that: score=0.8134  |  Anh gia: score=0.4210  |  Threshold=0.75

✓ PASS | Condition 4: Attendance Cooldown 5 phút
       Lan 1: Ghi OK  |  Lan 2 (< 5 phut): Bi tu choi

✓ PASS | Condition 5: Frame Skipping tăng FPS
       Khong skip: 10 lan xu ly nang  |  Co skip (=2): 5 lan  |  Giam 50% load

✓ PASS | Condition 6: Multi-face Support
       Gia lap 3 khuon mat dong thoi  |  Ten: Person_1, Person_2, Person_3

=================================================================
  KẾT QUẢ: 6/6 điều kiện PASS
  🎉 TẤT CẢ ĐIỀU KIỆN ĐỀU PASS!
=================================================================
```

### 4. Xem điểm danh & thống kê

```bash
# Xem điểm danh hôm nay
python main.py --mode attendance

# Xem thống kê tổng quan
python main.py --mode stats
```

### 5. Tùy chọn nâng cao

```bash
# Dùng GPU
python main.py --gpu

# Dùng camera số 1 (webcam ngoài)
python main.py --camera 1

# Model nhỏ hơn (buffalo_s) - nhanh hơn, kém chính xác hơn
python main.py --model buffalo_s

# Bật log chi tiết
python main.py --verbose
```

---

## Thông số kỹ thuật

| Thông số | Giá trị | Mô tả |
|---|---|---|
| Cosine threshold | 0.6 | Ngưỡng nhận diện khuôn mặt |
| Spoof threshold | 0.75 | Ngưỡng Anti-Spoofing (LIVE/SPOOF) |
| Attendance cooldown | 5 phút | Tránh điểm danh trùng |
| Frame skip | 2 | Xử lý 1/2 frame để tăng FPS |
| Input size | 640×480 | Resize trước khi detect |
| Face alignment | 112×112 | Chuẩn ArcFace |
| Embedding dim | 512-D | buffalo_l output |

---

## Kiến trúc Pipeline

```
Camera Frame
     │
     ▼ resize_frame()
  640×480 BGR
     │
     ▼ InsightFace.get()
  Face Detection (RetinaFace)
  Face Embedding (buffalo_l / 512-D)
     │
     ├──▶ Anti-Spoofing CNN
     │         │
     │    live_score ≥ 0.75?
     │       YES │  NO
     │           │   └──▶ [SPOOF] Vẽ hộp đỏ, KHÔNG điểm danh
     │           ▼
     │    Cosine Similarity vs Gallery
     │    max_sim ≥ 0.6?
     │       YES │  NO
     │           │   └──▶ [Unknown] Vẽ hộp đỏ
     │           ▼
     │    Attendance Cooldown Check (5 phút)
     │           │
     │           ▼
     │    SQLite: log_attendance()
     │           │
     │           ▼
     └──▶ [LIVE + Recognized] Vẽ hộp xanh, tên, confidence
```

---

## Màu sắc giao diện

| Màu | Ý nghĩa |
|---|---|
| 🟢 Xanh lá | Nhận diện thành công + Khuôn mặt thật (LIVE) |
| 🔴 Đỏ | Không nhận diện (Unknown) hoặc phát hiện giả mạo (SPOOF) |

---

## Anti-Spoofing

Module chống giả mạo hoạt động dựa trên 3 đặc trưng chính:

1. **LBP texture histogram** – Ảnh in/màn hình có texture phẳng hơn khuôn mặt thật
2. **FFT frequency analysis** – Ảnh giả thường thiếu tần số cao
3. **Laplacian variance (sharpness)** – Khuôn mặt thật có nhiều chi tiết cạnh hơn

Để tích hợp mô hình đã train sẵn (MiniFASNet, Silent-Face):
```python
from anti_spoofing import load_pretrained_weights, create_anti_spoofing

clf = create_anti_spoofing(threshold=0.75)
load_pretrained_weights(clf, "models/antispoofing_weights.npy")
```

---

## Xuất dữ liệu

```python
from database import DatabaseManager

db = DatabaseManager("attendance.db")

# Xuất điểm danh hôm nay ra CSV
db.export_to_csv("bao_cao_diem_danh.csv")

# Lấy điểm danh theo ngày
records = db.get_attendance_by_date("2024-01-15")
```

---

## Troubleshooting

| Vấn đề | Giải pháp |
|---|---|
| Không mở được camera | Kiểm tra `--camera 0` hoặc `--camera 1` |
| InsightFace không tải được model | Kiểm tra kết nối Internet; thử `insightface.model_zoo.get_model('buffalo_l')` |
| FPS thấp | Dùng `--model buffalo_s`; bật `--gpu`; tăng `FRAME_SKIP` trong code |
| ImportError: skimage | `pip install scikit-image` |
| Module not found | Đảm bảo đã kích hoạt venv và chạy `pip install -r requirements.txt` |

---

## Tài liệu tham khảo

- [InsightFace GitHub](https://github.com/deepinsight/insightface)
- [RetinaFace Paper](https://arxiv.org/abs/1905.00641)
- [ArcFace Paper](https://arxiv.org/abs/1801.07698)
- [Silent-Face Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)

---

*Phát triển trong khuôn khổ Khóa luận Tốt nghiệp – Trường Đại học Quy Nhơn – 2024*
