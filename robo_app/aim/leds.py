"""aim: leds."""

from .helpers import safe_call
from .settings import LED_COLORS, TOP_LED_REFRESH_S
from robomaster import blaster
from robomaster import led
import queue
import threading
import time

# ==========================================
# 8. ไฟ LED
# ==========================================
class LedManager:
    """ คุมไฟทั้งหมดของหุ่น
    - ไฟ Top (บน Gimbal) และไฟ Blaster: ดับตลอด กันแสงสะท้อนเข้ากล้อง
    - ไฟ Bottom (ฐานล้อ): แสดงสถานะด้วยสีขาว
        ติดค้าง = SAFE, กระพริบ = Auto-Fire พร้อมยิง, หายใจ = โหมดจูน

    คำสั่งไฟทุกคำสั่งส่งจาก thread แยก เพราะ set_led ต้องรอหุ่นตอบกลับ (Wi-Fi สะดุดรอได้ถึง 3 วินาที)
    ถ้าส่งจาก Loop หลัก ภาพและปุ่มจะค้างทุกครั้งที่ส่ง
    """
    _STOP = object()

    def __init__(self, ep_led, ep_blaster):
        self._led = ep_led
        self._blaster = ep_blaster
        self._bottom_state = None
        self._jobs: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="led-worker", daemon=True)
        self._thread.start()

    def _worker(self):
        next_top_off = 0.0
        while True:
            try:
                job = self._jobs.get(timeout=0.2)
            except queue.Empty:
                job = None
            if job is self._STOP:
                return
            if job is not None:
                job()
            if time.monotonic() >= next_top_off:  # ดับไฟ Top ซ้ำเป็นระยะ เผื่อ firmware เปิดเอง (เช่นตอนโดนยิง)
                self._top_off()
                next_top_off = time.monotonic() + TOP_LED_REFRESH_S

    def _top_off(self):
        safe_call("top LED off", self._led.set_led, comp=led.COMP_TOP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)

    def silence_gimbal_lights(self):
        def job():
            self._top_off()
            safe_call("blaster LED off", self._blaster.set_led, brightness=0, effect=blaster.LED_OFF)
        self._jobs.put(job)

    def show_status(self, color: str, armed: bool, tuning: bool):
        if tuning:
            effect = led.EFFECT_BREATH
        elif armed:
            effect = led.EFFECT_FLASH
        else:
            effect = led.EFFECT_ON
        state = (color, effect)
        if state == self._bottom_state:
            return  # ส่งคำสั่งเฉพาะตอนสถานะเปลี่ยน
        self._bottom_state = state
        r, g, b = LED_COLORS[color]
        self._jobs.put(lambda: safe_call("bottom LED", self._led.set_led, comp=led.COMP_BOTTOM_ALL,
                                         r=r, g=g, b=b, effect=effect, freq=2))

    def close(self):
        """ หยุด thread แล้วดับไฟทั้งหมด (ตอนปิดโปรแกรม) """
        self._jobs.put(self._STOP)
        self._thread.join(timeout=5)
        safe_call("all LED off", self._led.set_led, comp=led.COMP_ALL, r=0, g=0, b=0, effect=led.EFFECT_OFF)
        self._bottom_state = None


