"""
utils.py - Các hàm tiện ích dùng chung
Hệ thống Định danh Khuôn mặt Thời gian Thực
Trường Đại học Quy Nhơn - Khoa CNTT
"""

import cv2
import numpy as np
import time
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ====================================================================== #
#  XỬ LÝ HÌNH ẢNH                                                         #
# ====================================================================== #

def resize_frame(frame: np.ndarray, target_width: int = 640,
                 target_height: int = 480) -> np.ndarray:
    """
    Resize frame về kích thước chuẩn để tăng tốc xử lý.

    Args:
        frame:         Frame gốc từ camera
        target_width:  Chiều rộng đích (mặc định 640)
        target_height: Chiều cao đích (mặc định 480)

    Returns:
        Frame đã resize
    """
    h, w = frame.shape[:2]
    if w == target_width and h == target_height:
        return frame
    return cv2.resize(frame, (target_width, target_height),
                      interpolation=cv2.INTER_LINEAR)


def align_face(frame: np.ndarray, landmark: np.ndarray,
               output_size: int = 112) -> Optional[np.ndarray]:
    """
    Căn chỉnh (align) khuôn mặt dựa trên 5 điểm landmark.
    Sử dụng phép biến đổi Affine để chuẩn hóa vị trí mắt, mũi, miệng.

    Args:
        frame:       Frame gốc (BGR)
        landmark:    Mảng 5 điểm [(x,y)...] từ InsightFace
        output_size: Kích thước ảnh output (mặc định 112x112)

    Returns:
        Ảnh khuôn mặt đã căn chỉnh hoặc None nếu thất bại
    """
    # 5 điểm chuẩn (arcface standard template)
    dst_pts = np.array([
        [38.2946, 51.6963],   # mắt trái
        [73.5318, 51.5014],   # mắt phải
        [56.0252, 71.7366],   # mũi
        [41.5493, 92.3655],   # miệng trái
        [70.7299, 92.2041],   # miệng phải
    ], dtype=np.float32)

    # Scale theo output_size
    dst_pts = dst_pts * (output_size / 112.0)

    src_pts = landmark.astype(np.float32)

    try:
        # Tính ma trận biến đổi Affine (2D similarity transform)
        transform_matrix = _estimate_norm(src_pts, output_size)
        aligned = cv2.warpAffine(frame, transform_matrix,
                                 (output_size, output_size),
                                 flags=cv2.INTER_LINEAR)
        return aligned
    except Exception as e:
        logger.warning(f"Lỗi align face: {e}")
        return None


def _estimate_norm(lmk: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Ước tính ma trận biến đổi Similarity transform từ landmark.
    (Hàm nội bộ - được gọi bởi align_face)
    """
    from skimage import transform as trans

    # Template chuẩn (arcface)
    arcface_dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    arcface_dst = arcface_dst * image_size / 112

    tform = trans.SimilarityTransform()
    tform.estimate(lmk, arcface_dst)
    M = tform.params[:2]
    return M


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Tính độ tương đồng Cosine giữa hai vector embedding.

    Args:
        vec_a, vec_b: Vector đặc trưng khuôn mặt (đã normalize hoặc chưa)

    Returns:
        Giá trị trong [-1, 1], càng gần 1 càng giống nhau
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def preprocess_face_for_antispoofing(face_img: np.ndarray,
                                     size: int = 64) -> np.ndarray:
    """
    Tiền xử lý ảnh khuôn mặt cho module Anti-Spoofing CNN.

    Args:
        face_img: Ảnh khuôn mặt BGR (bất kỳ kích thước)
        size:     Kích thước đầu vào CNN (mặc định 64x64)

    Returns:
        Tensor chuẩn hóa shape (1, 3, size, size) - channel first
    """
    img = cv2.resize(face_img, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0

    # Chuẩn hóa theo ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # Chuyển từ HWC → CHW và thêm batch dimension
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


# ====================================================================== #
#  VẼ GIAO DIỆN OPENCV                                                    #
# ====================================================================== #

def draw_face_box(frame: np.ndarray,
                  bbox: Tuple[int, int, int, int],
                  name: str,
                  confidence: float,
                  is_live: bool,
                  is_recognized: bool) -> np.ndarray:
    """
    Vẽ bounding box và thông tin lên frame.

    Màu sắc:
        - Xanh lá (0,255,0):  Nhận diện thành công + LIVE
        - Đỏ (0,0,255):       Không nhận diện hoặc SPOOF

    Args:
        frame:          Frame BGR
        bbox:           (x1, y1, x2, y2) tọa độ bounding box
        name:           Tên người được nhận diện
        confidence:     Độ tin cậy (0-1)
        is_live:        True = khuôn mặt thật
        is_recognized:  True = đã nhận diện được

    Returns:
        Frame đã vẽ annotations
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Xác định màu: xanh nếu nhận diện OK & LIVE, đỏ nếu ngược lại
    if is_recognized and is_live:
        color = (0, 255, 0)      # Xanh lá
        status = "LIVE"
    elif not is_live:
        color = (0, 0, 255)      # Đỏ
        status = "SPOOF"
    else:
        color = (0, 0, 255)      # Đỏ
        status = "UNKNOWN"

    # Vẽ bounding box với độ dày 2px
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Vẽ góc trang trí (bo góc giả)
    corner_len = 15
    thickness  = 3
    _draw_corners(frame, x1, y1, x2, y2, color, corner_len, thickness)

    # Nhãn nền (background rectangle cho text)
    label_text = f"{name} {confidence:.2f}" if is_recognized else "Unknown"
    status_text = f"[{status}]"

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thick = 1

    # Đo kích thước text
    (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, text_thick)
    (sw, sh), _ = cv2.getTextSize(status_text, font, font_scale, text_thick)
    max_w = max(tw, sw)

    # Vẽ nền mờ cho text
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1 - th * 3 - 10), (x1 + max_w + 6, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Vẽ text tên + confidence
    cv2.putText(frame, label_text,
                (x1 + 3, y1 - th - 8),
                font, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA)

    # Vẽ text trạng thái LIVE / SPOOF
    status_color = (0, 255, 0) if is_live else (0, 0, 255)
    cv2.putText(frame, status_text,
                (x1 + 3, y1 - 4),
                font, font_scale, status_color, text_thick, cv2.LINE_AA)

    return frame


def _draw_corners(frame, x1, y1, x2, y2, color, length, thickness):
    """Vẽ 4 góc trang trí trên bounding box."""
    # Góc trên trái
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    # Góc trên phải
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    # Góc dưới trái
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    # Góc dưới phải
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)


def draw_hud(frame: np.ndarray, fps: float, face_count: int,
             attendance_count: int) -> np.ndarray:
    """
    Vẽ HUD (Heads-Up Display) ở góc trên trái màn hình.

    Hiển thị: FPS, số khuôn mặt phát hiện, số lượt điểm danh hôm nay.
    """
    h, w = frame.shape[:2]

    # Nền bán trong suốt ở góc trên trái
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    color = (200, 230, 255)
    thick = 1

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (8, 22), font, scale, color, thick, cv2.LINE_AA)
    cv2.putText(frame, f"Khuon mat: {face_count}",
                (8, 44), font, scale, color, thick, cv2.LINE_AA)
    cv2.putText(frame, f"Diem danh hom nay: {attendance_count}",
                (8, 66), font, scale, color, thick, cv2.LINE_AA)

    # Timestamp góc trên phải
    ts = time.strftime("%Y-%m-%d  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, font, 0.45, 1)
    cv2.putText(frame, ts, (w - tw - 8, 20),
                font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


def draw_instructions(frame: np.ndarray) -> np.ndarray:
    """Vẽ hướng dẫn phím tắt ở góc dưới màn hình."""
    h, w = frame.shape[:2]
    instructions = [
        "Q: Thoat  |  S: Chup & Dang ky  |  A: Xem diem danh  |  R: Tai lai DB"
    ]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    color = (150, 150, 150)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 25), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (6, h - 8 + i * 18),
                    font, scale, color, 1, cv2.LINE_AA)
    return frame


# ====================================================================== #
#  FPS COUNTER                                                             #
# ====================================================================== #

class FPSCounter:
    """Tính FPS theo cửa sổ trượt để tránh dao động mạnh."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps  = []

    def tick(self) -> float:
        """Gọi mỗi frame. Trả về FPS hiện tại."""
        now = time.time()
        self.timestamps.append(now)

        # Giữ chỉ `window_size` mốc thời gian gần nhất
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ====================================================================== #
#  LOGGING SETUP                                                           #
# ====================================================================== #

def setup_logging(level: int = logging.INFO):
    """Cấu hình logging chuẩn cho toàn hệ thống."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
