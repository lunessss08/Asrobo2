"""maze: vision."""

from .settings import (
    CIRCLE_MIN_CIRCULARITY,
    SHAPE_EPSILON_RATIO,
    SQUARE_ANGLE_TOL,
    SQUARE_SIDE_RATIO,
)
from robo_app import aim as gs
import cv2
import math
import numpy as np

def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


def classify_shape(contour, area: float) -> str:
    """ วงกลม / จัตุรัส / ผืนผ้า จากความกลม (4*pi*A/P^2) และจำนวนมุมของ approxPolyDP
    (ตรรกะเดียวกับ gimbalshootandrun.py; ที่นั่นเรียกผืนผ้าว่า para) """
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return "?"
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    approx = cv2.approxPolyDP(contour, SHAPE_EPSILON_RATIO * perimeter, True)
    if len(approx) >= 5 and circularity >= CIRCLE_MIN_CIRCULARITY:
        return "circle"
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float64)
        sides = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
        if min(sides) <= 1e-6:
            return "?"
        angles = []
        for i in range(4):
            v1, v2 = pts[(i - 1) % 4] - pts[i], pts[(i + 1) % 4] - pts[i]
            cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
            angles.append(math.degrees(math.acos(min(max(cosine, -1.0), 1.0))))
        square = (max(sides) / min(sides) <= SQUARE_SIDE_RATIO
                  and all(abs(a - 90.0) <= SQUARE_ANGLE_TOL for a in angles))
        return "square" if square else "rect"
    return "circle" if circularity >= CIRCLE_MIN_CIRCULARITY else "rect"


def plate_shape(fd, target, hsv_range, blind_zones) -> str:
    """ รูปทรงของป้ายหนึ่งแผ่น: คอนทัวร์ใหญ่สุดของสีนั้นในกรอบป้าย """
    mask = gs.build_mask(fd.hsv, hsv_range, blind_zones)
    x, y, bw, bh = target.bbox
    pad = 3
    roi = mask[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]
    contours = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    if not contours:
        return "?"
    c = max(contours, key=cv2.contourArea)
    return classify_shape(c, cv2.contourArea(c))


