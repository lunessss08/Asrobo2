"""aim: control."""

from .settings import (
    FIRE_COOLDOWN_S,
    LOCK_TOLERANCE_X_DEG,
    LOCK_TOLERANCE_Y_DEG,
    MIN_LOCK_HITS,
    PID_INTEGRAL_LIMIT,
    SETTLE_TIME_S,
)
from typing import Optional
from typing import Tuple
import time

# ==========================================
# 6. PID Controller
# ==========================================
class PIDController:
    """ PID ของ Gimbal หนึ่งแกน
    Input : error เป็นองศา (มุมระหว่างเป้ากับจุดเล็ง)
    Output: ความเร็วเชิงมุม deg/s สำหรับ gimbal.drive_speed()
    """
    STALE_DT_S = 0.5  # ห่างจากรอบก่อนนานเกินนี้ (เช่นหลัง recenter) ถือว่าเริ่มใหม่

    def __init__(self, kp, ki, kd, output_limit=25.0, integral_limit=PID_INTEGRAL_LIMIT,
                 deadband=0.0, d_smoothing=0.5):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.deadband = deadband
        self.d_smoothing = d_smoothing  # 0 = ไม่กรอง, ใกล้ 1 = กรองแรง (ภาพจากกล้องมี noise)
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._derivative = 0.0
        self._last_error = 0.0
        self._last_time: Optional[float] = None

    @staticmethod
    def _clamp(value, limit):
        return max(-limit, min(limit, value))

    def compute(self, error: float) -> float:
        now = time.monotonic()
        if abs(error) < self.deadband:
            error = 0.0

        if self._last_time is None or now - self._last_time > self.STALE_DT_S:
            # รอบแรกยังไม่มี dt ที่เชื่อถือได้: ใช้แค่ P กัน derivative kick ตอนเจอเป้าใหม่
            dt = 0.0
            self._derivative = 0.0
        else:
            dt = max(now - self._last_time, 1e-3)
            raw_d = (error - self._last_error) / dt
            self._derivative = self.d_smoothing * self._derivative + (1.0 - self.d_smoothing) * raw_d

        p = self.kp * error
        d = self.kd * self._derivative

        # Anti-windup: สะสม I เฉพาะตอนที่ output ยังไม่อิ่มตัว หรือการสะสมช่วยดึงออกจากจุดอิ่มตัว
        candidate = self._clamp(self._integral + error * dt, self.integral_limit)
        unsaturated = p + self.ki * candidate + d
        if abs(unsaturated) <= self.output_limit or unsaturated * error < 0:
            self._integral = candidate

        self._last_error = error
        self._last_time = now
        return self._clamp(p + self.ki * self._integral + d, self.output_limit)


# ==========================================
# 7. ลอจิกการยิง (Settling Time + Safety)
# ==========================================
class FireController:
    """ ตัดสินว่ายิงได้หรือยัง (เกณฑ์ล็อกจากโค้ดเพื่อน)
    ยิงเมื่อเห็นเป้าติดกันอย่างน้อย MIN_LOCK_HITS เฟรม และ |err_x| <= LOCK_TOLERANCE_X_DEG,
    |err_y| <= LOCK_TOLERANCE_Y_DEG ต่อเนื่องนานเกิน SETTLE_TIME_S
    และ Auto-Fire เปิด และไม่อยู่ในโหมดจูน และพ้น cooldown
    """

    def __init__(self):
        self.armed = False    # เปิดเฉพาะตอนสั่งเริ่มยิง
        self.settled = False  # เล็งนิ่งครบเวลาแล้วหรือยัง (ใช้ตอนซ้อมเล็งในภารกิจ)
        self._lock_start: Optional[float] = None
        self._hits = 0
        self._last_seen: Optional[float] = None
        self._last_fire = float("-inf")

    def reset_lock(self):
        self._lock_start = None
        self._hits = 0
        self._last_seen = None
        self.settled = False

    def evaluate(self, err_x: float, err_y: float, tuning: bool) -> Tuple[str, bool]:
        """ คืนค่า (ข้อความสถานะสำหรับ HUD, ควรยิงหรือไม่) เรียกเฉพาะเฟรมที่เห็นเป้า """
        now = time.monotonic()
        self.settled = False
        if self._last_seen is not None and now - self._last_seen > 0.5:
            self.reset_lock()
        self._last_seen = now
        aligned = abs(err_x) <= LOCK_TOLERANCE_X_DEG and abs(err_y) <= LOCK_TOLERANCE_Y_DEG
        if not aligned:
            self.reset_lock()
            return "TRACKING", False
        self._hits += 1
        if self._hits < MIN_LOCK_HITS:
            self._lock_start = None
            return "TRACKING", False

        if self._lock_start is None:
            self._lock_start = now
        held = now - self._lock_start
        if held <= SETTLE_TIME_S:
            return f"SETTLING {held:.2f}s", False
        self.settled = True
        if tuning:
            return "LOCKED - TUNING (FIRE BLOCKED)", False
        if not self.armed:
            return "LOCKED - SAFE", False
        if now - self._last_fire < FIRE_COOLDOWN_S:
            return "LOCKED - RELOADING", False
        return "FIRE", True

    def mark_fired(self):
        self._last_fire = time.monotonic()
        self.reset_lock()


