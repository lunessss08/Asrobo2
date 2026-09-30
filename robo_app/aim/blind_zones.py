"""aim: blind_zones."""

from .helpers import put_text
from .settings import MIN_ZONE_PX, Zone
from typing import List
from typing import Optional
from typing import Tuple
import cv2
import numpy as np

# ==========================================
# 4. พื้นที่ตาบอด (Blind Zones)
# ==========================================
class BlindZones:
    """ ถมสีดำลงบน Mask เพื่อให้ Vision มองไม่เห็นบริเวณนั้น
    - auto_zone : บังหัว Gimbal ของหุ่นเอง ใช้งานเสมอ
    - user_zones: กรอบที่ผู้ใช้ลากเมาส์วาดเพิ่มตอน Setup
    เก็บพิกัดเป็นสัดส่วน 0-1 จึงใช้ได้ทุกความละเอียดกล้อง
    """

    def __init__(self, auto_zone: Zone, user_zones: Optional[List[Zone]] = None):
        self.auto_zone = auto_zone
        self.user_zones: List[Zone] = list(user_zones or [])

    @staticmethod
    def to_pixels(zone: Zone, w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        x1, y1, x2, y2 = zone
        return (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))

    def all_zones(self) -> List[Zone]:
        return [self.auto_zone] + self.user_zones

    def apply(self, mask: np.ndarray):
        h, w = mask.shape[:2]
        for zone in self.all_zones():
            p1, p2 = self.to_pixels(zone, w, h)
            cv2.rectangle(mask, p1, p2, 0, thickness=-1)

    def draw(self, frame: np.ndarray, fill_alpha: float = 0.0):
        h, w = frame.shape[:2]
        if fill_alpha > 0:
            overlay = frame.copy()
            for zone in self.all_zones():
                p1, p2 = self.to_pixels(zone, w, h)
                cv2.rectangle(overlay, p1, p2, (0, 0, 0), thickness=-1)
            cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
        for i, zone in enumerate(self.all_zones()):
            p1, p2 = self.to_pixels(zone, w, h)
            label, color = ("AUTO BLIND ZONE", (0, 165, 255)) if i == 0 else (f"BLIND ZONE {i}", (180, 180, 180))
            cv2.rectangle(frame, p1, p2, color, 2)
            put_text(frame, label, (p1[0] + 8, p1[1] + 22), color, 0.5, 1)


class ZoneDrawer:
    """ Mouse callback ตอน Setup: กดค้าง-ลาก-ปล่อย = เพิ่มกรอบพื้นที่ตาบอด 1 กรอบ """

    def __init__(self, blind_zones: BlindZones):
        self.blind_zones = blind_zones
        self.frame_size = (1, 1)  # (w, h) อัปเดตทุกเฟรม
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_end: Optional[Tuple[int, int]] = None

    def on_mouse(self, event, x, y, flags, param):
        w, h = self.frame_size
        x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            (x1, y1), (x2, y2) = self.drag_start, (x, y)
            self.drag_start = self.drag_end = None
            if abs(x2 - x1) >= MIN_ZONE_PX and abs(y2 - y1) >= MIN_ZONE_PX:
                self.blind_zones.user_zones.append(
                    (min(x1, x2) / w, min(y1, y2) / h, max(x1, x2) / w, max(y1, y2) / h))
                print(f">> เพิ่มพื้นที่ตาบอด #{len(self.blind_zones.user_zones)}")


