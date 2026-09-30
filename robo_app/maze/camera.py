"""maze: camera."""

from typing import Optional
from typing import Tuple
import numpy as np
import threading
import time

class FrameGrabber(threading.Thread):
    """ อ่านกล้องใน thread เดียวตลอดเวลา ให้ทั้ง GUI และตัวคุมหุ่นหยิบภาพล่าสุดไปใช้
    (ถ้าอ่านกล้องหลายที่ จะแย่งเฟรมกันจนภาพกระตุก)
    """

    def __init__(self, io):
        super().__init__(name="frame-grabber", daemon=True)
        self.io = io
        self.frame: Optional[np.ndarray] = None
        self.seq = 0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            frame = self.io.read_frame()
            if frame is not None:
                self.frame, self.seq = frame, self.seq + 1
            if self.io.sim:
                time.sleep(1 / 30)

    def wait_new(self, after_seq: int, timeout: float = 1.5, check=None) -> Tuple[Optional[np.ndarray], int]:
        """ รอภาพที่ใหม่กว่า after_seq (ภาพหลังหันหัวเสร็จ ไม่ใช่ภาพเก่าที่ค้างอยู่) """
        t0 = time.monotonic()
        while self.seq <= after_seq and time.monotonic() - t0 < timeout:
            if check is not None:
                check()
            time.sleep(0.01)
        if check is not None:
            check()
        if self.seq <= after_seq:
            return None, after_seq  # กล้องไม่ส่งภาพใหม่: ห้ามเอาภาพเก่าที่ค้างไปตัดสิน
        return self.frame, self.seq

    def stop(self):
        self._stop_event.set()


