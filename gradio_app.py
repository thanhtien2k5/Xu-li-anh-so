"""
gradio_app.py - Ứng dụng Web nhận diện khuôn mặt với Gradio
Hệ thống Định danh Khuôn mặt Thời gian Thực - ĐH Quy Nhơn

Chạy: python gradio_app.py
"""

import gradio as gr
import cv2
import numpy as np
import logging
import threading
from datetime import datetime

# Import các module có sẵn
from face_recognition_system import FaceRecognitionSystem
from database import DatabaseManager
from utils import setup_logging

# Cấu hình logging (tắt bớt log để giao diện sạch)
setup_logging(logging.WARNING)

# ==================== KHỞI TẠO HỆ THỐNG TOÀN CỤC ====================
print(">>> Đang khởi tạo hệ thống nhận diện...")
system = FaceRecognitionSystem(
    db_path="attendance.db",
    model_name="buffalo_l",   # hoặc "buffalo_s" nếu muốn nhanh hơn
    use_gpu=True              # True nếu có GPU, False nếu dùng CPU
)
print(">>> Hệ thống sẵn sàng!")

# Biến lưu frame cuối cùng (để dùng khi đăng ký)
last_frame = None
frame_lock = threading.Lock()   # tránh xung đột khi đăng ký và xử lý frame

# ==================== XỬ LÝ TỪNG FRAME (STREAM) ====================
def process_frame(frame):
    """
    Hàm được Gradio gọi mỗi khi có frame mới từ webcam.
    - frame: numpy array (BGR)
    - Trả về: frame đã vẽ kết quả nhận diện (BGR)
    """
    global last_frame
    if frame is None:
        return None

    # Lưu frame hiện tại (dùng cho đăng ký)
    with frame_lock:
        last_frame = frame.copy()

    # Xử lý nhận diện (hàm process_frame bên trong system đã có resize, skip frame, vẽ bounding box)
    annotated, results = system.process_frame(frame)
    
    # (Tuỳ chọn) In ra console số khuôn mặt để debug
    # print(f"Frame: {len(results)} faces")
    
    return annotated

# ==================== CHỨC NĂNG ĐĂNG KÝ ====================
def register_from_last_frame(person_id, name, role):
    """
    Đăng ký khuôn mặt từ frame lưu gần nhất.
    Trả về thông báo kết quả.
    """
    global last_frame
    if last_frame is None:
        return "❌ Chưa có frame nào từ camera. Hãy đảm bảo webcam đang chạy."

    if not person_id or not name:
        return "❌ Vui lòng nhập đầy đủ Mã và Họ tên."

    with frame_lock:
        frame_copy = last_frame.copy()

    # Gọi hàm đăng ký có sẵn (đã có trong FaceRecognitionSystem)
    # Lưu ý: Hàm _register_from_frame yêu cầu nhập từ terminal, ta sẽ viết lại logic trực tiếp
    faces = system.face_app.get(frame_copy)
    if not faces:
        return "❌ Không phát hiện khuôn mặt trong frame hiện tại. Hãy nhìn thẳng vào webcam."

    # Lấy khuôn mặt lớn nhất
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    embedding = face.normed_embedding

    success = system.db.add_person(person_id, name, embedding, role)
    if success:
        system.reload_gallery()
        return f"✅ Đăng ký thành công: {name} ({person_id})"
    else:
        return "❌ Đăng ký thất bại. Kiểm tra database."

# ==================== CHỨC NĂNG RELOAD GALLERY ====================
def reload_gallery():
    system.reload_gallery()
    return "🔄 Đã tải lại danh sách người dùng từ database."

# ==================== CHỨC NĂNG XEM ĐIỂM DANH HÔM NAY ====================
def get_attendance_dataframe():
    records = system.db.get_today_attendance()
    if not records:
        # Trả về dataframe rỗng nhưng vẫn hiển thị cột
        return gr.Dataframe(value=[], headers=["person_id", "name", "confidence", "is_live", "timestamp"])
    
    # Chuyển thành list of list
    data = [[r["person_id"], r["name"], f"{r['confidence']:.2f}", "LIVE" if r["is_live"] else "SPOOF", r["timestamp"]] for r in records]
    headers = ["Mã số", "Họ tên", "Độ tin cậy", "Trạng thái", "Thời gian"]
    return gr.Dataframe(value=data, headers=headers)

# ==================== XÂY DỰNG GIAO DIỆN GRADIO ====================
with gr.Blocks(title="Nhận diện khuôn mặt - ĐH Quy Nhơn", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎓 Hệ thống Định danh Khuôn mặt Thời gian Thực
    **Trường Đại học Quy Nhơn - Khoa Công nghệ Thông tin**  
    *Công nghệ: InsightFace (RetinaFace + buffalo_l) | Anti‑Spoofing | SQLite*
    """)
    
    with gr.Row():
        # Cột trái: Video stream
        with gr.Column(scale=3):
            webcam_input = gr.Image(
                sources=["webcam"], 
                streaming=True,
                label="Camera trực tiếp",
                elem_id="webcam"
            )
            output_video = gr.Image(
                label="Kết quả nhận diện",
                streaming=True
            )
        
        # Cột phải: Các điều khiển
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Đăng ký khuôn mặt mới")
            person_id = gr.Textbox(label="Mã định danh (VD: SV001)", placeholder="Nhập mã...")
            person_name = gr.Textbox(label="Họ và tên", placeholder="Nhập tên...")
            role = gr.Radio(["student", "teacher", "staff"], label="Vai trò", value="student")
            register_btn = gr.Button("💾 Đăng ký từ frame hiện tại", variant="primary")
            register_status = gr.Textbox(label="Trạng thái", interactive=False)
            
            gr.Markdown("### 🔧 Hệ thống")
            reload_btn = gr.Button("🔄 Tải lại danh sách người dùng")
            reload_status = gr.Textbox(label="Trạng thái reload", interactive=False)
            
            gr.Markdown("### 📊 Điểm danh hôm nay")
            attendance_btn = gr.Button("📋 Xem điểm danh hôm nay")
            attendance_output = gr.Dataframe(label="Danh sách điểm danh", interactive=False)
    
    # Kết nối sự kiện
    # 1. Stream webcam -> xử lý frame -> output
    webcam_input.stream(
        fn=process_frame,
        inputs=webcam_input,
        outputs=output_video,
        stream_every=0.01  # càng nhỏ càng mượt, nhưng phụ thuộc vào tốc độ xử lý
    )
    
    # 2. Đăng ký
    register_btn.click(
        fn=register_from_last_frame,
        inputs=[person_id, person_name, role],
        outputs=register_status
    ).then(
        fn=lambda: gr.update(value=""),  # xóa thông báo cũ sau 3 giây (không bắt buộc)
        outputs=register_status,
        queue=False
    )
    
    # 3. Reload gallery
    reload_btn.click(
        fn=reload_gallery,
        inputs=[],
        outputs=reload_status
    )
    
    # 4. Xem điểm danh
    attendance_btn.click(
        fn=get_attendance_dataframe,
        inputs=[],
        outputs=attendance_output
    )

# ==================== CHẠY ỨNG DỤNG ====================
if __name__ == "__main__":
    # Chạy trên localhost, cổng 7860
    # Có thể thay share=True để tạo link công khai (ngrok)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)