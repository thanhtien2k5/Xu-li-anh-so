"""
database.py - Module quản lý cơ sở dữ liệu SQLite
Hệ thống Định danh Khuôn mặt Thời gian Thực
Trường Đại học Quy Nhơn - Khoa CNTT
"""

import sqlite3
import numpy as np
import pickle
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Quản lý toàn bộ thao tác với cơ sở dữ liệu SQLite:
    - Lưu trữ thông tin sinh viên / nhân viên
    - Lưu trữ vector đặc trưng khuôn mặt (face embedding)
    - Ghi nhận điểm danh (attendance) với kiểm tra trùng lặp
    """

    def __init__(self, db_path: str = "attendance.db"):
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()

    # ------------------------------------------------------------------ #
    #  KẾT NỐI & KHỞI TẠO                                                 #
    # ------------------------------------------------------------------ #

    def _connect(self):
        """Tạo kết nối đến file SQLite (tạo mới nếu chưa tồn tại)."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # trả kết quả dạng dict-like
        logger.info(f"Đã kết nối database: {self.db_path}")

    def _create_tables(self):
        """Tạo các bảng cần thiết nếu chưa tồn tại."""
        cursor = self.conn.cursor()

        # Bảng lưu thông tin người dùng
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   TEXT UNIQUE NOT NULL,   -- mã định danh (VD: SV001)
                name        TEXT NOT NULL,           -- họ tên
                role        TEXT DEFAULT 'student',  -- student / teacher / staff
                embedding   BLOB,                    -- vector đặc trưng (pickle numpy)
                created_at  TEXT DEFAULT (datetime('now','localtime')),
                updated_at  TEXT DEFAULT (datetime('now','localtime'))
            )
        """)

        # Bảng ghi nhận điểm danh
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id   TEXT NOT NULL,
                name        TEXT NOT NULL,
                confidence  REAL,                    -- độ tin cậy nhận diện (0-1)
                is_live     INTEGER DEFAULT 1,        -- 1=LIVE, 0=SPOOF
                timestamp   TEXT DEFAULT (datetime('now','localtime')),
                date        TEXT DEFAULT (date('now','localtime'))
            )
        """)

        # Tạo index để tăng tốc truy vấn
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_attendance_person_time
            ON attendance(person_id, timestamp)
        """)

        self.conn.commit()
        logger.info("Đã khởi tạo bảng database.")

    # ------------------------------------------------------------------ #
    #  QUẢN LÝ NGƯỜI DÙNG                                                  #
    # ------------------------------------------------------------------ #

    def add_person(self, person_id: str, name: str, embedding: np.ndarray,
                   role: str = "student") -> bool:
        """
        Thêm hoặc cập nhật thông tin người dùng cùng vector embedding.

        Args:
            person_id: Mã định danh (VD: 'SV001')
            name:      Họ và tên
            embedding: Vector đặc trưng khuôn mặt (numpy array 512-D)
            role:      Vai trò: 'student' | 'teacher' | 'staff'

        Returns:
            True nếu thành công
        """
        embedding_blob = pickle.dumps(embedding)
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO persons (person_id, name, role, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    name       = excluded.name,
                    role       = excluded.role,
                    embedding  = excluded.embedding,
                    updated_at = datetime('now','localtime')
            """, (person_id, name, role, embedding_blob))
            self.conn.commit()
            logger.info(f"Đã lưu người dùng: {person_id} - {name}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi lưu người dùng: {e}")
            return False

    def get_all_persons(self) -> list:
        """
        Lấy toàn bộ danh sách người dùng kèm embedding.

        Returns:
            List of dict: [{'person_id', 'name', 'role', 'embedding'}, ...]
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT person_id, name, role, embedding FROM persons")
        rows = cursor.fetchall()

        persons = []
        for row in rows:
            embedding = pickle.loads(row["embedding"]) if row["embedding"] else None
            persons.append({
                "person_id": row["person_id"],
                "name":      row["name"],
                "role":      row["role"],
                "embedding": embedding,
            })
        return persons

    def delete_person(self, person_id: str) -> bool:
        """Xóa người dùng khỏi database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))
            self.conn.commit()
            logger.info(f"Đã xóa người dùng: {person_id}")
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Lỗi khi xóa người dùng: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  QUẢN LÝ ĐIỂM DANH                                                   #
    # ------------------------------------------------------------------ #

    def log_attendance(self, person_id: str, name: str,
                       confidence: float, is_live: bool,
                       cooldown_minutes: int = 5) -> bool:
        """
        Ghi nhận điểm danh, tránh ghi trùng trong khoảng cooldown_minutes.

        Args:
            person_id:        Mã người dùng
            name:             Họ tên
            confidence:       Độ tin cậy nhận diện (0-1)
            is_live:          True nếu khuôn mặt thật (Anti-Spoofing pass)
            cooldown_minutes: Không ghi nếu đã điểm danh trong X phút

        Returns:
            True nếu ghi thành công, False nếu trùng hoặc lỗi
        """
        if not is_live:
            logger.warning(f"Phát hiện SPOOF - không ghi điểm danh: {name}")
            return False

        try:
            cursor = self.conn.cursor()

            # Kiểm tra xem đã điểm danh trong vòng cooldown_minutes chưa
            cutoff = (datetime.now() - timedelta(minutes=cooldown_minutes)
                      ).strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute("""
                SELECT id FROM attendance
                WHERE person_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT 1
            """, (person_id, cutoff))

            if cursor.fetchone():
                # Đã điểm danh trong khoảng cooldown → bỏ qua
                logger.debug(f"Bỏ qua điểm danh trùng: {name} (cooldown {cooldown_minutes} phút)")
                return False

            # Ghi điểm danh mới
            cursor.execute("""
                INSERT INTO attendance (person_id, name, confidence, is_live)
                VALUES (?, ?, ?, ?)
            """, (person_id, name, confidence, int(is_live)))
            self.conn.commit()
            logger.info(f"✓ Điểm danh: {name} ({person_id}) | conf={confidence:.3f}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Lỗi ghi điểm danh: {e}")
            return False

    def get_today_attendance(self) -> list:
        """Lấy danh sách điểm danh trong ngày hôm nay."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT person_id, name, confidence, is_live, timestamp
            FROM attendance
            WHERE date = date('now','localtime')
            ORDER BY timestamp DESC
        """)
        return [dict(row) for row in cursor.fetchall()]

    def get_attendance_by_date(self, date_str: str) -> list:
        """
        Lấy điểm danh theo ngày cụ thể.

        Args:
            date_str: Định dạng 'YYYY-MM-DD'
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT person_id, name, confidence, is_live, timestamp
            FROM attendance
            WHERE date = ?
            ORDER BY timestamp
        """, (date_str,))
        return [dict(row) for row in cursor.fetchall()]

    def get_attendance_stats(self) -> dict:
        """Thống kê tổng quan điểm danh."""
        cursor = self.conn.cursor()

        # Tổng số bản ghi
        cursor.execute("SELECT COUNT(*) as total FROM attendance")
        total = cursor.fetchone()["total"]

        # Số người điểm danh hôm nay
        cursor.execute("""
            SELECT COUNT(DISTINCT person_id) as cnt
            FROM attendance WHERE date = date('now','localtime')
        """)
        today_count = cursor.fetchone()["cnt"]

        # Số bản ghi SPOOF bị chặn (is_live = 0 không được ghi, nhưng ta vẫn log warning)
        cursor.execute("SELECT COUNT(*) as cnt FROM persons")
        total_persons = cursor.fetchone()["cnt"]

        return {
            "total_records":  total,
            "today_count":    today_count,
            "total_persons":  total_persons,
        }

    # ------------------------------------------------------------------ #
    #  TIỆN ÍCH                                                            #
    # ------------------------------------------------------------------ #

    def export_to_csv(self, output_path: str = "attendance_export.csv") -> bool:
        """Xuất dữ liệu điểm danh ra file CSV."""
        try:
            import csv
            records = self.get_today_attendance()
            if not records:
                logger.warning("Không có dữ liệu để xuất.")
                return False

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            logger.info(f"Đã xuất CSV: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Lỗi xuất CSV: {e}")
            return False

    def close(self):
        """Đóng kết nối database."""
        if self.conn:
            self.conn.close()
            logger.info("Đã đóng kết nối database.")

    def __del__(self):
        self.close()
