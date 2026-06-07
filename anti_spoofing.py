"""
anti_spoofing.py - Module chống giả mạo khuôn mặt (Face Anti-Spoofing)
Hệ thống Định danh Khuôn mặt Thời gian Thực
Trường Đại học Quy Nhơn - Khoa CNTT

Kiến trúc: CNN 5 tầng nhẹ (Lightweight CNN)
    Conv1 → Conv2 → Conv3 → Conv4 → Conv5 → FC → Sigmoid
    Input: 64×64×3  |  Output: xác suất LIVE (0-1)

Lưu ý: Trong thực tế production nên dùng mô hình đã train
        (MiniFASNet, Silent-Face, v.v.). Module này mô phỏng
        pipeline đầy đủ với weights giả lập.
"""

import cv2
import numpy as np
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)


# ====================================================================== #
#  ĐỊNH NGHĨA KIẾN TRÚC CNN (NumPy-based, không cần PyTorch/TF)           #
# ====================================================================== #

class ConvLayer:
    """Tầng Convolution + BatchNorm + ReLU đơn giản (NumPy)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        # Khởi tạo weights ngẫu nhiên theo He initialization
        fan_in = in_ch * kernel * kernel
        self.W = np.random.randn(out_ch, in_ch, kernel, kernel).astype(np.float32)
        self.W *= np.sqrt(2.0 / fan_in)
        self.b = np.zeros(out_ch, dtype=np.float32)
        # BatchNorm params
        self.gamma = np.ones(out_ch,  dtype=np.float32)
        self.beta  = np.zeros(out_ch, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (N, C, H, W)
        Output:  (N, out_ch, H', W')
        """
        N, C, H, W = x.shape
        k = self.kernel
        pad = k // 2
        H_out = H  # same padding
        W_out = W

        # Padding
        x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')

        # Convolution (im2col approach đơn giản)
        out = np.zeros((N, self.out_ch, H_out, W_out), dtype=np.float32)
        for oc in range(self.out_ch):
            for ic in range(self.in_ch):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = x_pad[:, ic, i:i+k, j:j+k]
                        out[:, oc, i, j] += np.sum(
                            patch * self.W[oc, ic], axis=(1,2)
                        )
            out[:, oc] += self.b[oc]

        # BatchNorm (inference mode: dùng mean/var của batch hiện tại)
        mean = out.mean(axis=(0,2,3), keepdims=True)
        var  = out.var( axis=(0,2,3), keepdims=True) + 1e-5
        out  = (out - mean) / np.sqrt(var)
        g    = self.gamma[np.newaxis, :, np.newaxis, np.newaxis]
        b    = self.beta[ np.newaxis, :, np.newaxis, np.newaxis]
        out  = out * g + b

        # ReLU
        return np.maximum(out, 0)


def _maxpool2d(x: np.ndarray, size: int = 2) -> np.ndarray:
    """Max Pooling 2×2 stride 2."""
    N, C, H, W = x.shape
    H2 = H // size
    W2 = W // size
    out = np.zeros((N, C, H2, W2), dtype=np.float32)
    for i in range(H2):
        for j in range(W2):
            patch = x[:, :, i*size:(i+1)*size, j*size:(j+1)*size]
            out[:, :, i, j] = patch.max(axis=(2,3))
    return out


def _global_avg_pool(x: np.ndarray) -> np.ndarray:
    """Global Average Pooling → (N, C)."""
    return x.mean(axis=(2,3))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


# ====================================================================== #
#  FAST INFERENCE (Dùng đặc trưng ảnh truyền thống khi CNN quá chậm)     #
# ====================================================================== #

class FastAntiSpoofFeatureExtractor:
    """
    Trích xuất đặc trưng nhanh bằng phân tích texture (LBP, FFT, gradient).
    Dùng khi muốn đạt >30 FPS mà không cần GPU.

    Cơ sở lý thuyết:
    - Ảnh in / màn hình thường có texture phẳng hơn khuôn mặt thật
    - Tần số cao (qua FFT) của ảnh giả thường thấp hơn
    - Gradient magnitude của ảnh thật phong phú hơn
    """

    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        """Trả về vector đặc trưng 12-D."""
        img = cv2.resize(face_bgr, (64, 64)).astype(np.float32)

        # 1. LBP texture features (Local Binary Pattern histogram)
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        lbp  = self._lbp(gray)
        hist, _ = np.histogram(lbp, bins=8, range=(0,256), density=True)

        # 2. FFT - phân tích tần số
        f     = np.fft.fft2(gray)
        fshift= np.fft.fftshift(f)
        mag   = np.log1p(np.abs(fshift))
        # Tỉ lệ năng lượng tần số cao vs thấp
        h, w  = mag.shape
        low   = mag[h//4:3*h//4, w//4:3*w//4].mean()
        high  = (mag.sum() - low * (h//2 * w//2)) / (h * w - h//2 * w//2 + 1e-6)
        fft_ratio = high / (low + 1e-6)

        # 3. Gradient magnitude
        gx    = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy    = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag  = np.sqrt(gx**2 + gy**2)
        g_mean= gmag.mean() / 255.0
        g_std = gmag.std()  / 255.0

        # 4. HSV saturation variance (ảnh in thường bão hòa màu thấp hơn)
        hsv   = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
        s_std = hsv[:,:,1].std() / 255.0

        feat = np.concatenate([hist, [fft_ratio, g_mean, g_std, s_std]])
        return feat.astype(np.float32)

    @staticmethod
    def _lbp(gray: np.ndarray) -> np.ndarray:
        """Tính LBP đơn giản với radius=1, 8 điểm lân cận."""
        h, w   = gray.shape
        result = np.zeros((h, w), dtype=np.uint8)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            shifted = np.roll(np.roll(gray, dy, 0), dx, 1)
            result  = (result << 1) | (gray >= shifted).astype(np.uint8)
        return result


# ====================================================================== #
#  ANTI-SPOOFING CLASSIFIER CHÍNH                                          #
# ====================================================================== #

class AntiSpoofingClassifier:
    """
    Classifier nhẹ dùng SVM-like linear decision boundary
    trên đặc trưng texture (Fast mode) hoặc CNN 5 tầng (Accurate mode).

    Threshold: xác suất >= threshold → LIVE, ngược lại → SPOOF
    """

    def __init__(self, threshold: float = 0.75, use_fast_mode: bool = True):
        """
        Args:
            threshold:      Ngưỡng xác suất để phân loại LIVE (mặc định 0.75)
            use_fast_mode:  True=dùng texture features (nhanh), False=CNN (chậm hơn)
        """
        self.threshold     = threshold
        self.use_fast_mode = use_fast_mode
        self.extractor     = FastAntiSpoofFeatureExtractor()

        # Trọng số tuyến tính đã huấn luyện sẵn (mô phỏng)
        # Trong thực tế: load từ file .npy sau khi train
        np.random.seed(42)
        self._weights = np.array([
             0.12,  0.08, -0.05,  0.15,  0.09,  0.11,  0.07,  0.06,  # LBP hist
            -0.25,  0.40,  0.35,  0.18                                  # FFT, grad, sat
        ], dtype=np.float32)
        self._bias = -0.10

        logger.info(f"AntiSpoofing khởi tạo | threshold={threshold} | "
                    f"mode={'FAST(texture)' if use_fast_mode else 'CNN'}")

    def predict(self, face_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Dự đoán khuôn mặt thật/giả.

        Args:
            face_bgr: Ảnh khuôn mặt BGR (bất kỳ kích thước)

        Returns:
            (is_live, confidence)
            - is_live:    True nếu khuôn mặt thật
            - confidence: Xác suất LIVE trong [0,1]
        """
        if face_bgr is None or face_bgr.size == 0:
            return False, 0.0

        try:
            if self.use_fast_mode:
                prob = self._predict_fast(face_bgr)
            else:
                prob = self._predict_cnn(face_bgr)

            is_live = prob >= self.threshold
            return is_live, float(prob)

        except Exception as e:
            logger.error(f"Lỗi Anti-Spoofing predict: {e}")
            return True, 1.0   # Fallback: coi là LIVE nếu lỗi

    def _predict_fast(self, face_bgr: np.ndarray) -> float:
        """
        Dự đoán nhanh dựa trên đặc trưng texture.
        Dùng hàm logistic trên combination tuyến tính.
        """
        feat   = self.extractor.extract(face_bgr)
        logit  = np.dot(self._weights, feat) + self._bias
        prob   = _sigmoid(np.array([logit]))[0]

        # Bổ sung heuristic: kiểm tra gradient phong phú
        gray   = cv2.cvtColor(
            cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_32F).var()

        # Khuôn mặt thật thường có Laplacian variance > 30 (rõ nét, nhiều chi tiết)
        sharpness_score = np.clip(laplacian_var / 500.0, 0, 1)

        # Kết hợp 2 điểm số
        final_prob = 0.6 * prob + 0.4 * sharpness_score
        return float(np.clip(final_prob, 0.0, 1.0))

    def _predict_cnn(self, face_bgr: np.ndarray) -> float:
        """
        Dự đoán qua CNN 5 tầng (chậm hơn, chính xác hơn về nguyên lý).
        Trong demo này dùng đặc trưng texture + noise injection để mô phỏng.
        """
        # Trong thực tế production: thay bằng model ONNX
        # ví dụ: session.run(None, {'input': preprocess(face_bgr)})
        base_prob = self._predict_fast(face_bgr)
        # Mô phỏng độ biến thiên nhỏ của CNN
        noise = np.random.normal(0, 0.02)
        return float(np.clip(base_prob + noise, 0.0, 1.0))

    def set_threshold(self, threshold: float):
        """Cập nhật ngưỡng phân loại."""
        self.threshold = np.clip(threshold, 0.0, 1.0)
        logger.info(f"Anti-Spoofing threshold mới: {self.threshold}")


# ====================================================================== #
#  HÀM TIỆN ÍCH PUBLIC                                                    #
# ====================================================================== #

def create_anti_spoofing(threshold: float = 0.75,
                         fast_mode: bool = True) -> AntiSpoofingClassifier:
    """
    Factory function - tạo Anti-Spoofing classifier.

    Args:
        threshold: Ngưỡng xác suất LIVE (0.0-1.0)
        fast_mode: Dùng chế độ nhanh (texture) hay CNN

    Returns:
        AntiSpoofingClassifier đã khởi tạo
    """
    return AntiSpoofingClassifier(threshold=threshold, use_fast_mode=fast_mode)


def load_pretrained_weights(classifier: AntiSpoofingClassifier,
                            weights_path: str) -> bool:
    """
    Tải trọng số đã huấn luyện từ file .npy.

    Args:
        classifier:   Instance AntiSpoofingClassifier
        weights_path: Đường dẫn đến file weights (.npy)

    Returns:
        True nếu tải thành công
    """
    if not os.path.exists(weights_path):
        logger.warning(f"Không tìm thấy file weights: {weights_path}")
        return False
    try:
        data = np.load(weights_path, allow_pickle=True).item()
        classifier._weights = data["weights"]
        classifier._bias    = data["bias"]
        logger.info(f"Đã tải weights: {weights_path}")
        return True
    except Exception as e:
        logger.error(f"Lỗi tải weights: {e}")
        return False
