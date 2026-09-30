"""aim: vision."""

from . import settings
from .blind_zones import BlindZones
from .color_config import HSVRange
from .helpers import put_text
from .settings import (
    BORDER_PX,
    CANNY_HIGH,
    CANNY_LOW,
    COLOR_ORDER,
    DEDUP_RATIO,
    DISPLAY_COLORS,
    MAX_CANDIDATES,
    MIN_AREA_RATIO,
    MIN_EXTENT,
    MIN_FILL,
    PROCESS_W,
    SIGN_ASPECT,
)
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import cv2
import numpy as np

# ==========================================
# 5. Vision: จับป้ายสี (ลอจิกจากโค้ดเพื่อน grimball_shoot_3targets/vision.py)
# ==========================================
@dataclass
class Target:
    color: str
    cx: float                         # จุดกลางกรอบป้าย (พิกเซลของภาพที่ประมวลผล)
    cy: float
    area: float
    bbox: Tuple[int, int, int, int]   # x, y, w, h


@dataclass
class FrameData:
    """ ข้อมูลที่คำนวณครั้งเดียวต่อเฟรม แล้วใช้ร่วมกันทุกสี """
    bgr: np.ndarray                   # ภาพขนาดที่ประมวลผล (พิกัดเป้าทั้งหมดอ้างอิงภาพนี้)
    hsv: np.ndarray
    edge_boxes: List[Tuple[float, Tuple[int, int, int, int]]]  # (พื้นที่, กรอบ) จากขอบภาพที่รูปร่างเป็นป้าย

    @property
    def size(self) -> Tuple[int, int]:
        h, w = self.hsv.shape[:2]
        return w, h


_MORPH_KERNEL = np.ones((3, 3), np.uint8)


def _sign_box(contour, w: int, h: int) -> Optional[Tuple[float, Tuple[int, int, int, int]]]:
    """ เกณฑ์รูปร่างป้าย (ไม่ขึ้นกับสี) คืน (พื้นที่, กรอบ) หรือ None ถ้าไม่เหมือนป้าย """
    x, y, bw, bh = cv2.boundingRect(contour)
    frame_area = w * h
    if bw * bh < MIN_AREA_RATIO * frame_area:
        return None  # เศษเล็กๆ ตัดทิ้งก่อนคำนวณอย่างอื่น (ภาพที่มีลายเยอะมีคอนทัวร์จากขอบเป็นพันเส้น)
    if bw > w * settings.MAX_SIGN_W or bh > h * settings.MAX_SIGN_H or not SIGN_ASPECT[0] <= bw / bh <= SIGN_ASPECT[1]:
        return None
    if x < BORDER_PX or y < BORDER_PX or x + bw >= w - BORDER_PX or y + bh >= h - BORDER_PX:
        return None  # ชิดขอบภาพ = เห็นป้ายไม่เต็มแผ่น
    area = cv2.contourArea(contour)
    if not MIN_AREA_RATIO * frame_area <= area <= settings.MAX_AREA_RATIO * frame_area:
        return None
    if area / (bw * bh) < MIN_EXTENT:
        return None  # ไม่ทึบเต็มกรอบ = ก้อนสีรูปร่างมั่ว
    return area, (x, y, bw, bh)


def prepare_frame(frame_bgr: np.ndarray) -> FrameData:
    """ ย่อภาพ (ถ้าใหญ่กว่า PROCESS_W) -> HSV และขอบภาพ ทำครั้งเดียวต่อเฟรม
    โค้ดเพื่อนคำนวณขอบภาพซ้ำทุกสี ที่นี่คำนวณครั้งเดียวและคัดรูปร่างป้ายไว้ก่อน จึงเร็วกว่ามาก
    """
    h, w = frame_bgr.shape[:2]
    small = frame_bgr
    if w > PROCESS_W:
        small = cv2.resize(frame_bgr, (PROCESS_W, int(round(h * PROCESS_W / w))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), CANNY_LOW, CANNY_HIGH)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    edge_boxes = []
    for contour in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]:
        box = _sign_box(contour, sw, sh)
        if box is not None:
            edge_boxes.append(box)
    return FrameData(small, hsv, edge_boxes)


def build_mask(hsv: np.ndarray, hsv_range: HSVRange, blind_zones: BlindZones) -> np.ndarray:
    mask = hsv_range.mask(hsv)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)   # ลบจุดสีเล็กๆ
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)  # อุดรูในป้าย
    blind_zones.apply(mask)  # ทำเป็นขั้นสุดท้าย พื้นที่ตาบอดจึงดำสนิทแน่นอน
    return mask


def find_targets(fd: FrameData, mask: np.ndarray, color: str) -> List[Target]:
    """ ป้ายสี color ทั้งหมดในภาพ เรียงจากใหญ่ไปเล็ก (สูงสุด MAX_CANDIDATES)
    ผู้สมัคร = คอนทัวร์จาก mask สีนั้น + คอนทัวร์จากขอบภาพ ที่รูปร่างเป็นป้าย
    แล้วข้างในกรอบ (ตัดขอบออก 1/5) ต้องเป็นสีนั้นอย่างน้อย MIN_FILL
    """
    w, h = fd.size
    boxes = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]:
        box = _sign_box(contour, w, h)
        if box is not None:
            boxes.append(box)
    boxes += fd.edge_boxes

    found: List[Target] = []
    for area, (x, y, bw, bh) in boxes:
        inset = max(2, min(bw, bh) // 5)
        interior = mask[y + inset:y + bh - inset, x + inset:x + bw - inset]
        if interior.size == 0 or cv2.countNonZero(interior) / interior.size < MIN_FILL:
            continue
        cx, cy = x + bw / 2.0, y + bh / 2.0
        if any(abs(cx - t.cx) < DEDUP_RATIO * w and abs(cy - t.cy) < DEDUP_RATIO * h for t in found):
            continue  # ป้ายเดียวกันที่เจอทั้งจาก mask และจากขอบภาพ
        found.append(Target(color, cx, cy, area, (x, y, bw, bh)))
    found.sort(key=lambda t: t.area, reverse=True)
    return found[:MAX_CANDIDATES]


def detect_all_colors(fd: FrameData, ranges: Dict[str, HSVRange], blind_zones: BlindZones) -> List[Target]:
    found: List[Target] = []
    for color in COLOR_ORDER:
        found.extend(find_targets(fd, build_mask(fd.hsv, ranges[color], blind_zones), color))
    return found


def draw_targets(frame: np.ndarray, targets: List[Target], scale: float = 1.0):
    """ scale = ขนาดภาพที่วาด / ขนาดภาพที่ประมวลผล """
    for t in targets:
        bgr = DISPLAY_COLORS[t.color]
        x, y, bw, bh = (int(round(v * scale)) for v in t.bbox)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), bgr, 2)
        cv2.circle(frame, (int(t.cx * scale), int(t.cy * scale)), 3, (255, 255, 255), -1)
        put_text(frame, t.color.upper(), (x, max(14, y - 6)), bgr, 0.5, 1)


