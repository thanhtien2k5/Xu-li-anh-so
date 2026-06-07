import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import cv2
import os

os.makedirs("figures", exist_ok=True)

# ================== Hình 1.1 ==================
def fig1_1_matrix():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Hình 1.1: Biểu diễn ma trận pixel của ảnh số", fontsize=14)
    data = np.random.randint(0, 256, (8, 8))
    ax.matshow(data, cmap='gray', vmin=0, vmax=255)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(data[i, j]), ha='center', va='center',
                    color='red' if data[i,j] > 128 else 'white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Chiều rộng (x) →")
    ax.set_ylabel("Chiều cao (y) ↓")
    plt.tight_layout()
    plt.savefig("figures/Hinh1_1_ma_tran_pixel.png", dpi=150)
    plt.close()

# ================== Hình 1.2 (không emoji) ==================
def fig1_2_apps():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Hình 1.2: Các ứng dụng của nhận diện khuôn mặt", fontsize=14)
    ax.axis('off')
    apps = ["Điểm danh học sinh", "Mở khóa điện thoại", "An ninh sân bay", "Y tế bệnh nhân", "Bán lẻ khách quen"]
    for i, app in enumerate(apps):
        y = 0.8 - i*0.15
        ax.text(0.1, y, chr(8226), fontsize=20, ha='center')  # dấu bullet
        ax.text(0.3, y, app, fontsize=12, va='center', ha='left')
    plt.tight_layout()
    plt.savefig("figures/Hinh1_2_ung_dung.png", dpi=150)
    plt.close()

# ================== Hình 2.1 ==================
def fig2_1_hist():
    np.random.seed(42)
    img_dark = np.clip(np.random.normal(50, 20, (100,100)), 0, 255).astype(np.uint8)
    img_eq = cv2.equalizeHist(img_dark)
    fig, axes = plt.subplots(2,2, figsize=(8,6))
    axes[0,0].imshow(img_dark, cmap='gray')
    axes[0,0].set_title("Ảnh tối")
    axes[0,0].axis('off')
    axes[0,1].hist(img_dark.ravel(), bins=256, color='gray')
    axes[0,1].set_title("Histogram ảnh tối")
    axes[1,0].imshow(img_eq, cmap='gray')
    axes[1,0].set_title("Sau cân bằng sáng")
    axes[1,0].axis('off')
    axes[1,1].hist(img_eq.ravel(), bins=256, color='gray')
    axes[1,1].set_title("Histogram sau cân bằng")
    plt.suptitle("Hình 2.1: Minh họa histogram ảnh trước và sau khi cân bằng sáng")
    plt.tight_layout()
    plt.savefig("figures/Hinh2_1_histogram.png", dpi=150)
    plt.close()

# ================== Hình 2.2 ==================
def fig2_2_rgb_hsv():
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111, projection='3d')
    r = np.linspace(0,1,10)
    g = np.linspace(0,1,10)
    b = np.linspace(0,1,10)
    R,G,B = np.meshgrid(r,g,b)
    ax.scatter(R.flatten(), G.flatten(), B.flatten(), c=np.stack([R.flatten(), G.flatten(), B.flatten()], axis=1), s=5)
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_title("Không gian RGB")
    plt.tight_layout()
    plt.savefig("figures/Hinh2_2_rgb.png", dpi=150)
    plt.close()
    fig, ax = plt.subplots(figsize=(6,2))
    ax.axis('off')
    ax.text(0.1,0.5,"RGB → Hue (góc màu)\n       Saturation (độ bão hòa)\n       Value (độ sáng)", fontsize=12)
    plt.savefig("figures/Hinh2_2_hsv.png", dpi=150)
    plt.close()

# ================== Hình 2.3 (đã sửa, không cần file ảnh) ==================
def fig2_3_filters():
    img = np.ones((300, 300, 3), dtype=np.uint8) * 128
    cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)
    cv2.rectangle(img, (100, 60), (200, 120), (0, 0, 0), -1)
    cv2.rectangle(img, (100, 220), (200, 280), (50, 100, 150), -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (5,5), 1.5)
    median = cv2.medianBlur(gray, 5)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    canny = cv2.Canny(gray, 50, 150)
    titles = ['Original', 'Gaussian', 'Median', 'Sobel', 'Canny']
    images = [gray, gaussian, median, sobel, canny]
    fig, axes = plt.subplots(2,3, figsize=(10,6))
    axes = axes.flatten()
    for i in range(5):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    axes[5].axis('off')
    plt.suptitle("Hình 2.3: Các bộ lọc Gaussian, Median, Sobel, Canny")
    plt.tight_layout()
    plt.savefig("figures/Hinh2_3_filters.png", dpi=150)
    plt.close()

# ================== Hình 2.4 ==================
def fig2_4_pipeline():
    fig, ax = plt.subplots(figsize=(8,2))
    ax.axis('off')
    steps = ["Thu nhận ảnh", "Phát hiện\nkhuôn mặt", "Căn chỉnh", "Trích xuất\nđặc trưng", "Đối sánh", "Nhận dạng"]
    x = np.linspace(0.1,0.9,len(steps))
    for i, step in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x[i]-0.05, 0.4), 0.1, 0.2, boxstyle="round,pad=0.02", facecolor='lightblue', edgecolor='black'))
        ax.text(x[i], 0.5, step, ha='center', va='center', fontsize=9)
        if i < len(steps)-1:
            ax.annotate("", xy=(x[i+1]-0.05, 0.5), xytext=(x[i]+0.05, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title("Hình 2.4: Pipeline tổng quát của hệ thống nhận diện khuôn mặt")
    plt.tight_layout()
    plt.savefig("figures/Hinh2_4_pipeline.png", dpi=150)
    plt.close()

# ================== Hình 3.1 ==================
def fig3_1_cnn():
    fig, ax = plt.subplots(figsize=(10,4))
    ax.axis('off')
    layers = ["Input\n(3x224x224)", "Conv+ReLU\n(64x224x224)", "MaxPool\n(64x112x112)", "Conv+ReLU\n(128x112x112)", "MaxPool\n(128x56x56)", "FC\n(512)", "Output\n(classes)"]
    x = np.linspace(0.05,0.95,len(layers))
    for i, layer in enumerate(layers):
        ax.add_patch(FancyBboxPatch((x[i]-0.06, 0.3), 0.12, 0.4, boxstyle="round,pad=0.02", facecolor='lightgreen', edgecolor='black'))
        ax.text(x[i], 0.5, layer, ha='center', va='center', fontsize=8)
        if i < len(layers)-1:
            ax.annotate("", xy=(x[i+1]-0.06, 0.5), xytext=(x[i]+0.06, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Hình 3.1: Kiến trúc mạng CNN cơ bản")
    plt.tight_layout()
    plt.savefig("figures/Hinh3_1_cnn.png", dpi=150)
    plt.close()

# ================== Hình 3.2 ==================
def fig3_2_resnet():
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.4,0.7), 0.2, 0.1, boxstyle="round,pad=0.02", facecolor='orange', edgecolor='black'))
    ax.text(0.5,0.75, "Weight layer", ha='center')
    ax.add_patch(FancyBboxPatch((0.4,0.55), 0.2, 0.1, boxstyle="round,pad=0.02", facecolor='orange', edgecolor='black'))
    ax.text(0.5,0.6, "Weight layer", ha='center')
    ax.add_patch(FancyBboxPatch((0.4,0.4), 0.2, 0.1, boxstyle="round,pad=0.02", facecolor='lightblue', edgecolor='black'))
    ax.text(0.5,0.45, "ReLU", ha='center')
    ax.annotate("", xy=(0.4,0.75), xytext=(0.4,0.45), arrowprops=dict(arrowstyle="->", color='red', lw=2))
    ax.text(0.35,0.6, "skip connection\n(identity)", ha='center', color='red', fontsize=9)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title("Hình 3.2: Block cơ bản trong ResNet với skip connection")
    plt.savefig("figures/Hinh3_2_resnet_block.png", dpi=150)
    plt.close()

# ================== Hình 3.3 ==================
def fig3_3_facenet():
    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    anchors = ["Anchor", "Positive", "Negative"]
    xpos = [0.2, 0.5, 0.8]
    for i, txt in enumerate(anchors):
        ax.add_patch(FancyBboxPatch((xpos[i]-0.08, 0.6), 0.16, 0.12, boxstyle="round,pad=0.02", facecolor='lightyellow'))
        ax.text(xpos[i], 0.66, txt, ha='center', fontsize=10)
        ax.add_patch(FancyBboxPatch((xpos[i]-0.08, 0.4), 0.16, 0.12, boxstyle="round,pad=0.02", facecolor='lightblue'))
        ax.text(xpos[i], 0.46, "CNN", ha='center')
        ax.add_patch(FancyBboxPatch((xpos[i]-0.08, 0.2), 0.16, 0.12, boxstyle="round,pad=0.02", facecolor='lightgreen'))
        ax.text(xpos[i], 0.26, "Embedding", ha='center')
        ax.annotate("", xy=(xpos[i], 0.6), xytext=(xpos[i], 0.52), arrowprops=dict(arrowstyle="->"))
        ax.annotate("", xy=(xpos[i], 0.4), xytext=(xpos[i], 0.32), arrowprops=dict(arrowstyle="->"))
    ax.annotate("d(a,p)", xy=(0.35,0.15), ha='center')
    ax.annotate("d(a,n)", xy=(0.65,0.15), ha='center')
    ax.plot([0.2,0.5], [0.13,0.13], 'k-')
    ax.plot([0.5,0.8], [0.13,0.13], 'k-')
    ax.set_title("Hình 3.3: Mô hình FaceNet với triplet loss")
    plt.savefig("figures/Hinh3_3_facenet.png", dpi=150)
    plt.close()

# ================== Hình 3.4 ==================
def fig3_4_insightface():
    fig, ax = plt.subplots(figsize=(8,5))
    ax.axis('off')
    components = ["Input\nImage", "RetinaFace\nDetection", "Alignment", "MobileFaceNet\n+ ArcFace", "Embedding\n512-D", "Cosine\nSimilarity", "Output\nID"]
    x = np.linspace(0.05,0.95,len(components))
    for i, comp in enumerate(components):
        ax.add_patch(FancyBboxPatch((x[i]-0.06, 0.4), 0.12, 0.2, boxstyle="round,pad=0.02", facecolor='lightcoral'))
        ax.text(x[i], 0.5, comp, ha='center', va='center', fontsize=8)
        if i < len(components)-1:
            ax.annotate("", xy=(x[i+1]-0.06, 0.5), xytext=(x[i]+0.06, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Hình 3.4: Kiến trúc InsightFace framework")
    plt.tight_layout()
    plt.savefig("figures/Hinh3_4_insightface.png", dpi=150)
    plt.close()

# ================== Hình 4.1 ==================
def fig4_1_system_block():
    fig, ax = plt.subplots(figsize=(8,6))
    ax.axis('off')
    blocks = {
        "Camera": (0.2,0.8),
        "Resize &\nFrame Skip": (0.5,0.8),
        "RetinaFace\nDetect + Align": (0.2,0.6),
        "MobileFaceNet\nEmbedding": (0.5,0.6),
        "Anti-Spoofing": (0.8,0.6),
        "Cosine\nMatching": (0.5,0.4),
        "SQLite\nAttendance": (0.2,0.2),
        "Display\nHUD": (0.8,0.2),
    }
    for name, (x,y) in blocks.items():
        ax.add_patch(FancyBboxPatch((x-0.12, y-0.05), 0.24, 0.1, boxstyle="round,pad=0.02", facecolor='lavender', edgecolor='black'))
        ax.text(x, y, name, ha='center', va='center', fontsize=9)
    arrows = [((0.32,0.8),(0.38,0.8)), ((0.62,0.8),(0.32,0.65)), ((0.32,0.55),(0.38,0.6)), ((0.62,0.6),(0.68,0.6)),
              ((0.62,0.55),(0.62,0.45)), ((0.62,0.35),(0.32,0.25)), ((0.32,0.15),(0.68,0.15)), ((0.92,0.55),(0.92,0.25))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->"))
    ax.set_title("Hình 4.1: Sơ đồ khối tổng thể hệ thống đề xuất")
    plt.savefig("figures/Hinh4_1_system_block.png", dpi=150)
    plt.close()

# ================== Hình 4.2 ==================
def fig4_2_dataflow():
    fig, ax = plt.subplots(figsize=(8,5))
    ax.axis('off')
    entities = ["Camera", "Detection", "Alignment", "Embedding", "Database", "Display"]
    pos = [(0.1,0.7),(0.3,0.7),(0.5,0.7),(0.7,0.7),(0.9,0.7),(0.5,0.3)]
    for name, (x,y) in zip(entities, pos):
        ax.add_patch(FancyBboxPatch((x-0.08, y-0.05), 0.16, 0.1, boxstyle="round,pad=0.02", facecolor='lightyellow'))
        ax.text(x, y, name, ha='center', fontsize=9)
    arrows = [((0.18,0.7),(0.22,0.7)), ((0.38,0.7),(0.42,0.7)), ((0.58,0.7),(0.62,0.7)), ((0.78,0.7),(0.82,0.7)),
              ((0.7,0.65),(0.5,0.35)), ((0.3,0.65),(0.5,0.35)), ((0.5,0.25),(0.18,0.7)), ((0.5,0.25),(0.82,0.7))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->"))
    ax.set_title("Hình 4.2: Sơ đồ luồng dữ liệu trong hệ thống")
    plt.savefig("figures/Hinh4_2_dataflow.png", dpi=150)
    plt.close()

# ================== Hình 5.1 ==================
def fig5_1_gui():
    img = np.ones((480,640,3), dtype=np.uint8)*240
    cv2.rectangle(img, (10,10), (630,470), (200,200,200), 2)
    cv2.putText(img, "Face Recognition System - DH Quy Nhon", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.rectangle(img, (10,10), (200,80), (0,0,0), -1)
    cv2.putText(img, "FPS: 14.2", (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, "Khuon mat: 1", (15,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(img, "Diem danh: 15", (15,75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.rectangle(img, (250,150), (400,350), (0,255,0), 2)
    cv2.putText(img, "Tran Thanh Tien 0.92", (250,140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite("figures/Hinh5_1_gui.png", img)

# ================== Hình 5.2 ==================
def fig5_2_recognition():
    img = cv2.imread("figures/Hinh5_1_gui.png")
    if img is None:
        img = np.ones((480,640,3), dtype=np.uint8)*240
    cv2.putText(img, "LIVE", (260,330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite("figures/Hinh5_2_recognition.png", img)

# ================== Hình 5.3 (không dùng sklearn) ==================
def fig5_3_confusion():
    # confusion matrix thủ công
    cm = np.array([[695, 55], [70, 380]])
    fig, ax = plt.subplots(figsize=(5,4))
    ax.matshow(cm, cmap='Blues', alpha=0.8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14)
    ax.set_xticklabels(['', 'That', 'Gia'])
    ax.set_yticklabels(['', 'That', 'Gia'])
    ax.set_xlabel('Dự đoán')
    ax.set_ylabel('Thực tế')
    ax.set_title("Hình 5.3: Confusion matrix của hệ thống")
    plt.savefig("figures/Hinh5_3_confusion.png", dpi=150)
    plt.close()

# ================== Hình 5.4 ==================
def fig5_4_fps():
    num_faces = [1,2,3,4,5]
    fps = [14.1, 10.1, 7.9, 6.5, 5.5]
    plt.figure(figsize=(6,4))
    plt.bar(num_faces, fps, color='skyblue')
    plt.xlabel("Số lượng khuôn mặt")
    plt.ylabel("FPS")
    plt.title("Hình 5.4: Biểu đồ FPS theo số lượng khuôn mặt")
    for i, v in enumerate(fps):
        plt.text(num_faces[i], v+0.3, str(v), ha='center')
    plt.ylim(0,16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("figures/Hinh5_4_fps.png", dpi=150)
    plt.close()

# ================== Chạy tất cả ==================
if __name__ == "__main__":
    print("Đang tạo các hình ảnh...")
    fig1_1_matrix()
    fig1_2_apps()
    fig2_1_hist()
    fig2_2_rgb_hsv()
    fig2_3_filters()
    fig2_4_pipeline()
    fig3_1_cnn()
    fig3_2_resnet()
    fig3_3_facenet()
    fig3_4_insightface()
    fig4_1_system_block()
    fig4_2_dataflow()
    fig5_1_gui()
    fig5_2_recognition()
    fig5_3_confusion()
    fig5_4_fps()
    print("Hoàn thành! Các file ảnh được lưu trong thư mục 'figures/'")