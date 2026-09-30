"""aim: helpers."""

from .settings import FONT, FOV_X, FOV_Y, FRAME_TIMEOUT_S
from typing import Optional
from typing import Tuple
import cv2
import numpy as np
import queue

# ==========================================
# 3. Helper ทั่วไป
# ==========================================
def safe_call(label: str, fn, *args, **kwargs):
    """ เรียกคำสั่ง SDK โดยไม่ให้โปรแกรมล่ม ถ้าการสื่อสารกับหุ่นสะดุดชั่วคราว """
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # SDK โยน exception ได้หลายแบบเมื่อ Wi-Fi มีปัญหา
        print(f"[WARN] {label}: {e}")
        return None


def read_frame(ep_camera) -> Optional[np.ndarray]:
    """ อ่านเฟรมล่าสุด; SDK โยน queue.Empty เมื่อหมดเวลา (ไม่ได้คืนค่า None) """
    try:
        return ep_camera.read_cv2_image(timeout=FRAME_TIMEOUT_S, strategy="newest")
    except queue.Empty:
        return None


def key_char(key: int) -> str:
    """ แปลงค่าจาก cv2.waitKey เป็นตัวอักษรพิมพ์เล็ก (กด Caps Lock ก็ยังใช้ได้) """
    return chr(key).lower() if key < 128 else ""


def put_text(img, text, org, color=(255, 255, 255), scale=0.6, thickness=2):
    """ เขียนตัวหนังสือมีขอบดำ อ่านง่ายทุกพื้นหลัง (putText รองรับเฉพาะ ASCII) """
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def pixel_to_degrees(px: float, py: float, w: int, h: int) -> Tuple[float, float]:
    """ แปลงพิกัดพิกเซล -> มุม (องศา) เทียบกับกึ่งกลางภาพ

    ใช้องศาต่อพิกเซลคงที่ (FOV / ความกว้างภาพ) ซึ่งสอดคล้องกับค่า FOV 96x54
    ที่เป็นสัดส่วน 16:9 พอดี (เลนส์มุมกว้างที่มี distortion ให้มุมเกือบแปรผันตรงกับระยะพิกเซล)
    X: + = เป้าอยู่ทางขวา   Y: + = เป้าอยู่ด้านบน (แกน y ของภาพชี้ลง จึงกลับเครื่องหมาย)
    """
    deg_x = (px - w / 2.0) * (FOV_X / w)
    deg_y = (h / 2.0 - py) * (FOV_Y / h)
    return deg_x, deg_y


def degrees_to_pixel(deg_x: float, deg_y: float, w: int, h: int) -> Tuple[int, int]:
    """ แปลงกลับจากองศา -> พิกเซล (ใช้วาดจุดเล็งบนจอ) """
    return int(round(w / 2.0 + deg_x * w / FOV_X)), int(round(h / 2.0 - deg_y * h / FOV_Y))


