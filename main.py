"""
main.py - File chạy chính
Hệ thống Định danh Khuôn mặt Thời gian Thực
Trường Đại học Quy Nhơn - Khoa Công nghệ Thông tin

Sinh viên: Trần Thanh Tiến  |  MSSV: 4651050270
GVHD:      Lê Thị Kim Nga

Cách chạy:
    python main.py                         # Chạy nhận diện trực tiếp
    python main.py --mode register         # Đăng ký khuôn mặt từ camera
    python main.py --mode register_img     # Đăng ký từ file ảnh
    python main.py --mode evaluate         # Đánh giá 6 điều kiện
    python main.py --mode attendance       # Xem điểm danh hôm nay
    python main.py --camera 1              # Dùng camera số 1
    python main.py --gpu                   # Dùng GPU (CUDA)
"""

import argparse
import logging
import sys
import os

from utils import setup_logging
from face_recognition_system import FaceRecognitionSystem, evaluate_condition


def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Hệ thống Định danh Khuôn mặt Thời gian Thực - ĐH Quy Nhơn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py                                 Chạy nhận diện
  python main.py --mode evaluate                 Kiểm tra 6 điều kiện
  python main.py --mode register                 Đăng ký qua camera
  python main.py --mode register_img \\
      --image face.jpg --id SV001 --name "Nguyen Van A"
  python main.py --mode attendance               Xem điểm danh hôm nay
        """
    )

    parser.add_argument("--mode", type=str, default="run",
                        choices=["run", "register", "register_img",
                                 "evaluate", "attendance", "stats"],
                        help="Chế độ chạy (mặc định: run)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Index camera (mặc định: 0)")
    parser.add_argument("--db", type=str, default="attendance.db",
                        help="Đường dẫn file SQLite (mặc định: attendance.db)")
    parser.add_argument("--model", type=str, default="buffalo_l",
                        choices=["buffalo_l", "buffalo_s"],
                        help="Model InsightFace (mặc định: buffalo_l)")
    parser.add_argument("--gpu", action="store_true",
                        help="Sử dụng GPU (CUDA)")
    parser.add_argument("--verbose", action="store_true",
                        help="Bật log chi tiết (DEBUG)")

    # Tham số đăng ký từ ảnh
    parser.add_argument("--image", type=str, default="",
                        help="Đường dẫn ảnh để đăng ký (dùng với --mode register_img)")
    parser.add_argument("--id", type=str, default="",
                        help="Mã định danh người dùng (VD: SV001)")
    parser.add_argument("--name", type=str, default="",
                        help="Họ và tên người dùng")
    parser.add_argument("--role", type=str, default="student",
                        choices=["student", "teacher", "staff"],
                        help="Vai trò (mặc định: student)")

    return parser.parse_args()


def print_banner():
    """In banner khởi động."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║         HỆ THỐNG ĐỊNH DANH KHUÔN MẶT THỜI GIAN THỰC         ║
║          Trường Đại học Quy Nhơn - Khoa CNTT                 ║
║          Bộ môn Trí tuệ Nhân tạo                             ║
╠══════════════════════════════════════════════════════════════╣
║  Công nghệ:  InsightFace + Anti-Spoofing + SQLite            ║
║  Model:      RetinaFace (detect) + buffalo_l (embed)         ║
║  Phím tắt:   Q=Thoát | S=Đăng ký | A=Điểm danh | R=Reload  ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def mode_run(system: FaceRecognitionSystem, args):
    """Chế độ nhận diện trực tiếp từ camera."""
    print("\n[MODE: RUN] Bắt đầu nhận diện khuôn mặt...")
    print(f"  Camera: {args.camera} | Model: {args.model}")
    print(f"  Nhấn Q hoặc ESC để thoát\n")
    system.run(camera_id=args.camera)


def mode_register(system: FaceRecognitionSystem, args):
    """Chế độ đăng ký khuôn mặt qua camera."""
    print("\n[MODE: REGISTER] Đăng ký khuôn mặt mới qua camera")

    person_id = args.id or input("Nhập mã định danh (VD: SV4651050270): ").strip()
    name      = args.name or input("Nhập họ và tên đầy đủ: ").strip()
    role      = args.role

    if not person_id or not name:
        print("Lỗi: Thiếu mã hoặc tên! Hủy đăng ký.")
        return

    print(f"\nĐăng ký: {name} ({person_id}) | role={role}")
    print("Nhìn thẳng vào camera, giữ khuôn mặt ổn định...")

    success = system.register_face_from_camera(
        person_id  = person_id,
        name       = name,
        role       = role,
        camera_id  = args.camera,
        num_samples= 5
    )

    if success:
        print(f"\n✓ Đăng ký thành công: {name} ({person_id})")
    else:
        print("\n✗ Đăng ký thất bại. Kiểm tra camera và thử lại.")


def mode_register_img(system: FaceRecognitionSystem, args):
    """Chế độ đăng ký từ file ảnh."""
    print("\n[MODE: REGISTER_IMG] Đăng ký từ file ảnh")

    image_path = args.image or input("Nhập đường dẫn ảnh: ").strip()
    person_id  = args.id   or input("Nhập mã định danh: ").strip()
    name       = args.name or input("Nhập họ và tên: ").strip()

    if not all([image_path, person_id, name]):
        print("Lỗi: Thiếu thông tin!")
        return

    success = system.register_face_from_image(
        image_path = image_path,
        person_id  = person_id,
        name       = name,
        role       = args.role
    )

    if success:
        print(f"\n✓ Đăng ký thành công: {name} từ {image_path}")
    else:
        print("\n✗ Đăng ký thất bại.")


def mode_evaluate(system: FaceRecognitionSystem, args):
    """Chế độ đánh giá 6 điều kiện hệ thống."""
    print("\n[MODE: EVALUATE] Kiểm tra 6 điều kiện hệ thống...")
    results = evaluate_condition(system)

    # Tóm tắt
    passed = sum(1 for v in results.values() if v["passed"])
    total  = len(results)
    print(f"\nTỉ lệ pass: {passed}/{total} ({passed/total*100:.0f}%)")


def mode_attendance(system: FaceRecognitionSystem, args):
    """Xem danh sách điểm danh hôm nay."""
    print("\n[MODE: ATTENDANCE] Danh sách điểm danh hôm nay:")
    system._show_attendance()


def mode_stats(system: FaceRecognitionSystem, args):
    """Xem thống kê tổng quan."""
    stats = system.db.get_attendance_stats()
    print("\n" + "="*50)
    print("  THỐNG KÊ HỆ THỐNG")
    print("="*50)
    print(f"  Tổng người dùng đã đăng ký: {stats['total_persons']}")
    print(f"  Tổng bản ghi điểm danh:     {stats['total_records']}")
    print(f"  Điểm danh hôm nay:          {stats['today_count']} người")
    print("="*50 + "\n")


def main():
    args = parse_args()

    # Cấu hình logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # Hiển thị banner
    print_banner()

    # Khởi tạo hệ thống
    try:
        system = FaceRecognitionSystem(
            db_path    = args.db,
            model_name = args.model,
            use_gpu    = args.gpu
        )
    except Exception as e:
        print(f"\n✗ Lỗi khởi tạo hệ thống: {e}")
        print("  → Kiểm tra lại: pip install -r requirements.txt")
        print("  → Đảm bảo InsightFace đã download model 'buffalo_l'")
        sys.exit(1)

    # Routing theo mode
    mode_map = {
        "run":          mode_run,
        "register":     mode_register,
        "register_img": mode_register_img,
        "evaluate":     mode_evaluate,
        "attendance":   mode_attendance,
        "stats":        mode_stats,
    }

    try:
        mode_map[args.mode](system, args)
    except KeyboardInterrupt:
        print("\n\nĐã dừng bởi người dùng.")
    except Exception as e:
        logging.error(f"Lỗi không mong đợi: {e}", exc_info=True)
    finally:
        system.db.close()
        print("\nĐã đóng hệ thống. Tạm biệt!")


if __name__ == "__main__":
    main()
