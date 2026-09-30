"""aim: widgets."""

from .blind_zones import ZoneDrawer
from .settings import FONT, UI_BTN, UI_TEXT, VIEW_H, VIEW_W
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
import cv2
import numpy as np

# ==========================================
# 9. GUI หน้าต่างเดียว (ปุ่ม/slider วาดเองด้วย OpenCV คลิกเมาส์ได้ และมีปุ่มลัดคีย์บอร์ด)
# ==========================================
def text_color_for(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """ ตัวหนังสือดำบนพื้นสว่าง (เช่นปุ่มสีเหลือง) และขาวบนพื้นเข้ม """
    b, g, r = bg
    return (20, 20, 20) if 0.114 * b + 0.587 * g + 0.299 * r > 150 else UI_TEXT


def label(canvas, text, x, y, color=UI_TEXT, scale=0.5, thickness=1, max_w: Optional[int] = None):
    """ เขียนข้อความบนแผง GUI ถ้ากำหนด max_w จะย่อตัวอักษรจนพอดีความกว้าง """
    if max_w is not None:
        while scale > 0.35 and cv2.getTextSize(text, FONT, scale, thickness)[0][0] > max_w:
            scale -= 0.03
    cv2.putText(canvas, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def dim(color: Tuple[int, int, int], factor: float = 0.35) -> Tuple[int, int, int]:
    return tuple(int(c * factor) for c in color)


@lru_cache(maxsize=4)
def hue_bar(w: int, h: int) -> np.ndarray:
    """ แถบสีรุ้งใต้ slider ของ H ให้เห็นว่าค่า Hue แต่ละช่วงคือสีอะไร """
    hues = np.tile(np.linspace(0, 179, w).astype(np.uint8), (h, 1))
    full = np.full((h, w), 255, np.uint8)
    return cv2.cvtColor(np.dstack([hues, full, full]), cv2.COLOR_HSV2BGR)


@dataclass
class Button:
    x: int
    y: int
    w: int
    h: int
    text: str
    action: str
    color: Tuple[int, int, int] = UI_BTN
    enabled: bool = True
    selected: bool = False

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def draw(self, canvas: np.ndarray, hover: bool):
        color = self.color if self.enabled else (55, 55, 55)
        if hover and self.enabled:
            color = tuple(min(255, c + 35) for c in color)
        cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), color, -1)
        if self.selected:
            cv2.rectangle(canvas, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), 2)
        big = self.h >= 50
        scale, thickness = (0.62, 2) if big else (0.5, 1)
        while scale > 0.35 and cv2.getTextSize(self.text, FONT, scale, thickness)[0][0] > self.w - 10:
            scale -= 0.03
        (tw, th), _ = cv2.getTextSize(self.text, FONT, scale, thickness)
        fg = text_color_for(color) if self.enabled else (110, 110, 110)
        cv2.putText(canvas, self.text, (self.x + (self.w - tw) // 2, self.y + (self.h + th) // 2),
                    FONT, scale, fg, thickness, cv2.LINE_AA)


@dataclass
class Slider:
    x: int
    y: int
    w: int
    text: str
    value: int
    max_value: int
    on_change: Callable[[int], None]
    hue: bool = False
    TRACK_DY = 20   # ระยะจากบรรทัดชื่อลงมาถึงราง slider
    TRACK_H = 14

    def contains(self, px: int, py: int) -> bool:
        ty = self.y + self.TRACK_DY
        return self.x - 8 <= px <= self.x + self.w + 8 and ty - 10 <= py <= ty + self.TRACK_H + 10

    def set_from_x(self, px: int):
        value = int(round((px - self.x) / self.w * self.max_value))
        value = min(max(value, 0), self.max_value)
        if value != self.value:
            self.value = value
            self.on_change(value)

    def draw(self, canvas: np.ndarray):
        ty = self.y + self.TRACK_DY
        label(canvas, f"{self.text}: {self.value}", self.x, self.y + 12, UI_TEXT, 0.5)
        if self.hue:
            canvas[ty:ty + self.TRACK_H, self.x:self.x + self.w] = hue_bar(self.w, self.TRACK_H)
        else:
            cv2.rectangle(canvas, (self.x, ty), (self.x + self.w, ty + self.TRACK_H), (85, 85, 85), -1)
            fill = self.x + int(self.value / self.max_value * self.w)
            cv2.rectangle(canvas, (self.x, ty), (fill, ty + self.TRACK_H), (160, 160, 160), -1)
        kx = self.x + int(self.value / self.max_value * self.w)
        cv2.rectangle(canvas, (kx - 4, ty - 4), (kx + 4, ty + self.TRACK_H + 4), (255, 255, 255), -1)
        cv2.rectangle(canvas, (kx - 4, ty - 4), (kx + 4, ty + self.TRACK_H + 4), (0, 0, 0), 1)


class Ui:
    """ เก็บปุ่ม/slider ของเฟรมล่าสุด (วาดใหม่ทุกเฟรม) และแปลงการคลิกเมาส์เป็น action
    action ถูกเก็บเข้าคิว แล้วให้ Loop หลักเป็นคนทำ (ไม่สั่งหุ่นจากใน mouse callback)
    """

    def __init__(self):
        self.buttons: List[Button] = []
        self.sliders: List[Slider] = []
        self.actions = deque()
        self.mouse = (-1, -1)
        self.zone_drawer: Optional[ZoneDrawer] = None  # เปิดใช้เฉพาะหน้า SETUP
        self._dragging: Optional[Slider] = None

    def begin(self):
        self.buttons, self.sliders = [], []

    def button(self, canvas, x, y, w, h, text, action, color=UI_BTN, enabled=True, selected=False):
        b = Button(x, y, w, h, text, action, color, enabled, selected)
        b.draw(canvas, b.contains(*self.mouse))
        self.buttons.append(b)

    def slider(self, canvas, x, y, w, text, value, max_value, on_change, hue=False):
        s = Slider(x, y, w, text, value, max_value, on_change, hue)
        s.draw(canvas)
        self.sliders.append(s)

    def on_mouse(self, event, x, y, flags, param):
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for b in self.buttons:
                if b.enabled and b.contains(x, y):
                    self.actions.append(b.action)
                    return
            for s in self.sliders:
                if s.contains(x, y):
                    self._dragging = s
                    s.set_from_x(x)
                    return
            if self.zone_drawer is not None and x < VIEW_W and y < VIEW_H:
                self.zone_drawer.on_mouse(event, x, y, flags, param)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._dragging is not None:
                self._dragging.set_from_x(x)
            elif self.zone_drawer is not None:
                self.zone_drawer.on_mouse(event, x, y, flags, param)
        elif event == cv2.EVENT_LBUTTONUP:
            if self._dragging is not None:
                self._dragging = None
            elif self.zone_drawer is not None:
                self.zone_drawer.on_mouse(event, x, y, flags, param)


