"""aim: color_config."""

from .settings import COLOR_ORDER, DEFAULT_HSV, Zone
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
import cv2
import json
import numpy as np

# ==========================================
# 2. ช่วงสี HSV และการบันทึกค่า
# ==========================================
@dataclass
class HSVRange:
    """ ช่วงสี HSV หนึ่งชุด (OpenCV: H 0-179, S/V 0-255)
    รองรับ Hue วนรอบ (h_min > h_max) ทำให้สีแดงใช้ Trackbar ชุดเดียวจูนได้ครบทั้งสองฝั่ง
    """
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    def __post_init__(self):
        # บีบค่าให้อยู่ในช่วงที่ OpenCV รับได้ กันค่าเพี้ยนจากไฟล์ config
        for f in fields(self):
            limit = 179 if f.name.startswith("h") else 255
            setattr(self, f.name, int(min(max(int(getattr(self, f.name)), 0), limit)))

    @property
    def wraps(self) -> bool:
        return self.h_min > self.h_max

    def _in_range(self, hsv: np.ndarray, h_lo: int, h_hi: int) -> np.ndarray:
        lower = np.array([h_lo, self.s_min, self.v_min], dtype=np.uint8)
        upper = np.array([h_hi, self.s_max, self.v_max], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        if not self.wraps:
            return self._in_range(hsv, self.h_min, self.h_max)
        return cv2.bitwise_or(self._in_range(hsv, self.h_min, 179), self._in_range(hsv, 0, self.h_max))

    def __str__(self) -> str:
        wrap = " (wrap)" if self.wraps else ""
        return f"H {self.h_min}-{self.h_max}{wrap}  S {self.s_min}-{self.s_max}  V {self.v_min}-{self.v_max}"


def default_color_ranges() -> Dict[str, HSVRange]:
    return {name: HSVRange(*values) for name, values in DEFAULT_HSV.items()}


def load_config(path: Path) -> Tuple[Dict[str, HSVRange], List[Zone], List[str], bool]:
    """ โหลดค่าที่ calibrate ไว้: HSV, พื้นที่ตาบอด และลำดับยิงล่าสุด
    ค่าสุดท้าย (found) เป็น False ถ้าไม่มีไฟล์หรือไฟล์เสีย = ต้อง calibrate ใหม่
    """
    ranges = default_color_ranges()
    if not path.exists():
        return ranges, [], list(COLOR_ORDER), False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, values in data.get("hsv", {}).items():
            if name in ranges:
                ranges[name] = HSVRange(**{f.name: values[f.name] for f in fields(HSVRange)})
        zones = []
        for zone in data.get("user_blind_zones", []):
            x1, y1, x2, y2 = (float(v) for v in zone)
            if 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0:
                zones.append((x1, y1, x2, y2))
        order = [c for c in data.get("fire_order", COLOR_ORDER) if c in ranges]
        order = list(dict.fromkeys(order))  # ตัดสีซ้ำ คงลำดับเดิม
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"[CONFIG] ⚠️ อ่าน {path.name} ไม่ได้ ({e}) ใช้ค่าเริ่มต้นแทน")
        return default_color_ranges(), [], list(COLOR_ORDER), False
    print(f"[CONFIG] โหลดค่า HSV และพื้นที่ตาบอด {len(zones)} กรอบ จาก {path.name}")
    return ranges, zones, order, True


def save_config(path: Path, ranges: Dict[str, HSVRange], zones: List[Zone], order: List[str]):
    data = {
        "hsv": {name: asdict(rng) for name, rng in ranges.items()},
        "user_blind_zones": [[round(v, 4) for v in zone] for zone in zones],
        "fire_order": list(order),
    }
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[CONFIG] 💾 บันทึกค่าลง {path.name}")
    except OSError as e:
        print(f"[CONFIG] ⚠️ บันทึก {path.name} ไม่ได้: {e}")


