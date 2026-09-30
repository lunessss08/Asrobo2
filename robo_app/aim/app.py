"""aim: app."""

from .blind_zones import BlindZones, ZoneDrawer
from .color_config import HSVRange, load_config, save_config
from .control import FireController, PIDController
from .helpers import degrees_to_pixel, key_char, put_text, read_frame, safe_call
from .leds import LedManager
from .mission import Mission
from .settings import (
    AIM_OFFSET_X_DEG,
    AIM_OFFSET_Y_DEG,
    AUTO_BLIND_ZONE,
    CANVAS_H,
    CANVAS_W,
    COLOR_ORDER,
    COLOR_SHORT,
    CONFIG_PATH,
    CONN_TYPE,
    DEFAULT_HSV,
    DISPLAY_COLORS,
    FOV_X,
    FOV_Y,
    KEY_TO_COLOR,
    LOCK_TOLERANCE_X_DEG,
    LOCK_TOLERANCE_Y_DEG,
    NUDGE_KEYS,
    PANEL_W,
    PITCH_PID,
    RECENTER_TIMEOUT_S,
    ROBOT_MODE,
    SCREEN_MISSION,
    SCREEN_READY,
    SCREEN_SETUP,
    SCREEN_TUNE,
    START_KEY_GUARD_S,
    STREAM_RESOLUTION,
    STRIP_H,
    UI_BG,
    UI_BLUE,
    UI_DIM,
    UI_GREEN,
    UI_OK,
    UI_ORANGE,
    UI_PANEL,
    UI_RED,
    UI_TEXT,
    VIEW_H,
    VIEW_W,
    WINDOW_NAME,
    YAW_PID,
)
from .vision import (
    FrameData,
    Target,
    build_mask,
    detect_all_colors,
    draw_targets,
    find_targets,
    prepare_frame,
)
from .widgets import Ui, dim, label, text_color_for
from dataclasses import replace
from robomaster import blaster
from robomaster import robot
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
import cv2
import numpy as np
import textwrap
import time

# ==========================================
# 11. แอปหลัก
# ==========================================
class AutoAimApp:
    """ หน้าต่างเดียว 4 หน้าจอ: SETUP -> TUNE (= calibrate) , READY <-> MISSION """

    def __init__(self, calibrate: bool = False):
        self.color_ranges, user_zones, self.fire_order, found = load_config(CONFIG_PATH)
        self.blind_zones = BlindZones(AUTO_BLIND_ZONE, user_zones)
        # รันครั้งแรก (ยังไม่มีค่าที่ calibrate ไว้) ต้อง calibrate ก่อนเสมอ
        self.calibrate = calibrate or not found
        if not found:
            print("[CONFIG] ยังไม่มีค่าสีที่ calibrate ไว้ -> เข้าโหมด CALIBRATE อัตโนมัติ")
        elif not calibrate:
            print(f"[CONFIG] ใช้ค่าที่ calibrate ไว้ (calibrate ใหม่: กดปุ่ม CALIBRATE ALL "
                  f"หรือรัน python gimbalshoot.py --calibrate)")

        self.screen = SCREEN_READY
        self.mission: Optional[Mission] = None
        self.ui = Ui()
        self.running = True
        self.tune_tab = 0              # หน้า TUNE: 0-3 = สีตาม COLOR_ORDER, 4 = ตรวจรวมทุกสี
        self._calibrating = False      # อยู่ในขั้น calibrate เต็ม (SETUP -> TUNE)
        self._order_fresh = True       # คลิกสีแรกหลังเข้าหน้า READY = เริ่มเรียงลำดับใหม่
        self._ready_since = 0.0
        self._last_result: List[str] = []

        self.pid_yaw = PIDController(**YAW_PID)
        self.pid_pitch = PIDController(**PITCH_PID)
        self.fire_ctrl = FireController()

        self.ep_robot = None
        self.ep_gimbal = None
        self.ep_camera = None
        self.ep_blaster = None
        self.leds: Optional[LedManager] = None
        self._stream_started = False

        self._fps = 0.0
        self._last_frame_time: Optional[float] = None
        self._fire_banner_until = 0.0

    @property
    def is_tuning(self) -> bool:
        """ อยู่ในหน้า calibrate/จูนสี = ห้ามยิงเด็ดขาด """
        return self.screen in (SCREEN_SETUP, SCREEN_TUNE)

    # ---------- วงจรชีวิตโปรแกรม ----------
    def run(self):
        try:
            if not self._connect():
                return
            if self.calibrate:
                self._enter_setup()                # มี recenter อยู่ในนี้
            else:
                self._enter_ready(recenter=True)   # recenter ก่อนเริ่ม Loop หลักเสมอ
            self._main_loop()
        except KeyboardInterrupt:
            print("\n>> ได้รับ Ctrl+C")
        finally:
            self._shutdown()

    def _connect(self) -> bool:
        print("=== กำลังเชื่อมต่อ RoboMaster... ===")
        self.ep_robot = robot.Robot()
        if not self.ep_robot.initialize(conn_type=CONN_TYPE):
            print("❌ เชื่อมต่อหุ่นไม่สำเร็จ ตรวจสอบ Wi-Fi / CONN_TYPE")
            return False

        self.ep_gimbal = self.ep_robot.gimbal
        self.ep_camera = self.ep_robot.camera
        self.ep_blaster = self.ep_robot.blaster
        safe_call("set robot mode", self.ep_robot.set_robot_mode, mode=ROBOT_MODE)

        # ดับไฟ Top + Blaster ก่อนเปิดกล้อง แล้วใช้แค่ไฟฐานล้อแสดงสถานะ
        self.leds = LedManager(self.ep_robot.led, self.ep_blaster)
        self.leds.silence_gimbal_lights()
        self._update_status_led()

        self._recenter("หลังเชื่อมต่อ")
        self.ep_camera.start_video_stream(display=False, resolution=STREAM_RESOLUTION)
        self._stream_started = True
        return True

    def _shutdown(self):
        print("\n=== กำลังปิดระบบอย่างปลอดภัย... ===")
        self.fire_ctrl.armed = False  # ปลดอาวุธก่อนทำอย่างอื่น
        if self.mission is not None:
            self.mission.print_summary(aborted=True)
            self.mission = None
        if self.is_tuning:
            print("[CONFIG] ออกระหว่าง calibrate/จูนสี: ค่าที่ยังไม่ได้กด SAVE จะไม่ถูกบันทึก")

        if self.ep_gimbal is not None:
            safe_call("gimbal stop", self.ep_gimbal.drive_speed, pitch_speed=0, yaw_speed=0)
            action = safe_call("gimbal recenter", self.ep_gimbal.recenter)
            if action is not None:
                safe_call("recenter wait", action.wait_for_completed, timeout=RECENTER_TIMEOUT_S)
        if self.leds is not None:
            self.leds.close()
        if self._stream_started:
            safe_call("stop video", self.ep_camera.stop_video_stream)
        if self.ep_robot is not None:
            safe_call("robot close", self.ep_robot.close)
        cv2.destroyAllWindows()
        print("=== ปิดระบบเรียบร้อย ===")

    # ---------- Gimbal ----------
    def _reset_aim(self):
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.fire_ctrl.reset_lock()

    def _stop_gimbal(self):
        safe_call("gimbal stop", self.ep_gimbal.drive_speed, pitch_speed=0, yaw_speed=0)
        self._reset_aim()

    def _recenter(self, reason: str):
        print(f"[GIMBAL] Recenter ({reason})")
        safe_call("gimbal stop", self.ep_gimbal.drive_speed, pitch_speed=0, yaw_speed=0)
        action = safe_call("gimbal recenter", self.ep_gimbal.recenter)
        if action is not None:
            safe_call("recenter wait", action.wait_for_completed, timeout=RECENTER_TIMEOUT_S)
        self._reset_aim()

    def _fire(self) -> bool:
        # ตรวจซ้ำอีกชั้นก่อนลั่นไก: ยิงได้เฉพาะระหว่างภารกิจ ไม่อยู่ในหน้าจูนสี และระบบยิงเปิดอยู่
        if self.screen != SCREEN_MISSION or self.is_tuning or not self.fire_ctrl.armed:
            return False
        safe_call("gimbal stop", self.ep_gimbal.drive_speed, pitch_speed=0, yaw_speed=0)
        print(f"🔥 ยิงเป้า {self.mission.plan[self.mission.index].color.upper()}")
        safe_call("blaster fire", self.ep_blaster.fire, fire_type=blaster.WATER_FIRE, times=1)
        # ไม่สั่งไฟ Blaster กระพริบ; ดับไฟซ้ำเผื่อ firmware เปิดเองตอนยิง
        self.leds.silence_gimbal_lights()
        self.fire_ctrl.mark_fired()
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self._fire_banner_until = time.monotonic() + 0.3
        return True

    # ---------- เปลี่ยนหน้าจอ ----------
    def _enter_setup(self):
        self.screen = SCREEN_SETUP
        self._calibrating = True
        self.ui.zone_drawer = ZoneDrawer(self.blind_zones)
        self.ui.zone_drawer.frame_size = (VIEW_W, VIEW_H)
        self._recenter("เริ่ม calibrate")
        self._update_status_led()
        print(">> 🛠️ CALIBRATE 1/2: ลากเมาส์บนภาพกล้องเพื่อบังสิ่งที่ไม่ใช่เป้า แล้วกด NEXT")

    def _enter_tune(self, calibrating: bool):
        self.screen = SCREEN_TUNE
        self._calibrating = calibrating
        self.ui.zone_drawer = None
        self.tune_tab = 0
        self._stop_gimbal()
        self._update_status_led()
        print(">> 🎨 จูนสี: ลาก slider จนช่อง MASK PREVIEW เห็นเฉพาะเป้าสีนั้น แล้วกด NEXT / SAVE")

    def _finish_tuning(self):
        self._save()
        print(">> ✅ บันทึกค่าสีและพื้นที่ตาบอดแล้ว")
        self._enter_ready(recenter=True)

    def _cancel_calibration(self):
        self.color_ranges, zones, _, _ = load_config(CONFIG_PATH)
        self.blind_zones.user_zones = zones
        print(">> ยกเลิก (ใช้ค่าเดิมที่บันทึกไว้)")
        self._enter_ready(recenter=True)

    def _enter_ready(self, recenter: bool = False):
        """ กลับหน้า READY: ปลดระบบยิง หยุดหัว แล้วรอเลือกลำดับยิงรอบใหม่ """
        self.screen = SCREEN_READY
        self.mission = None
        self._calibrating = False
        self.ui.zone_drawer = None
        if self.fire_ctrl.armed:
            self.fire_ctrl.armed = False
            print(">> Auto-Fire: OFF (SAFE)")
        if recenter:
            self._recenter("กลับหน้า READY")
        else:
            self._stop_gimbal()
        self._order_fresh = True
        self._ready_since = time.monotonic()
        self._update_status_led()
        self._print_order()

    # ---------- ภารกิจ ----------
    def _start_mission(self, fire: bool):
        if not self.fire_order:
            print(">> ⚠️ ยังไม่ได้เลือกสีที่จะยิง")
            return
        if time.monotonic() - self._ready_since < START_KEY_GUARD_S:
            return  # ไม่รับ ENTER ที่กดซ้ำ/ค้างมาจากหน้าก่อน กันเริ่มยิงโดยไม่ตั้งใจ

        self._save()  # จำลำดับยิงไว้เป็นค่าเริ่มต้นของรอบหน้า
        self._stop_gimbal()
        self.fire_ctrl.armed = fire
        self.screen = SCREEN_MISSION
        self.mission = Mission(self, self.fire_order)
        print(f">> 🚀 {'เริ่มยิง' if fire else 'ซ้อมเล็ง (ไม่ยิง)'} ตามลำดับ {self._order_text()} "
              f"| กวาดหาเป้า {len(self.mission.poses)} มุม")
        self._update_status_led()

    def _mission_result(self, aborted: bool) -> List[str]:
        m = self.mission
        fired = sum(t.status == "FIRED" for t in m.plan)
        head = f"Last mission{' (STOPPED)' if aborted else ''}: fired {fired}/{len(m.plan)} target(s)"
        if not m.plan:
            head += " - stopped while scanning" if m.phase == "SCAN" else " - no target of the chosen colors"
        return [head] + m.progress_lines(per_line=8)

    def _abort_mission(self):
        self._last_result = self._mission_result(aborted=True)
        self.mission.print_summary(aborted=True)
        self._enter_ready()

    def _finish_mission(self):
        self._last_result = self._mission_result(aborted=False)
        self.mission.print_summary()
        self._enter_ready(recenter=True)

    def _toggle_auto_fire(self):
        self.fire_ctrl.armed = not self.fire_ctrl.armed
        self.fire_ctrl.reset_lock()  # ต้องนิ่งครบ SETTLE_TIME_S ใหม่หลังเปิดระบบยิง
        print(f">> Auto-Fire: {'ON ⚠️' if self.fire_ctrl.armed else 'OFF (SAFE) - ซ้อมเล็งต่อจนจบ'}")
        self._update_status_led()

    # ---------- ลำดับยิง ----------
    def _toggle_order(self, color: str):
        if self._order_fresh:
            self.fire_order = []  # คลิกสีแรกหลังเข้าหน้า READY = เริ่มเรียงลำดับใหม่
            self._order_fresh = False
        if color in self.fire_order:
            self.fire_order.remove(color)  # คลิกสีที่เลือกไว้แล้ว = เอาออก
        else:
            self.fire_order.append(color)
        self._print_order()

    def _undo_order(self):
        self._order_fresh = False
        if self.fire_order:
            self.fire_order.pop()
        self._print_order()

    def _clear_order(self):
        self._order_fresh = False
        self.fire_order = []
        self._print_order()

    def _order_text(self, sep: str = " -> ") -> str:
        return sep.join(c.upper() for c in self.fire_order)

    def _print_order(self):
        print(f">> ลำดับยิง: {self._order_text() or '(ยังไม่ได้เลือก)'}")

    # ---------- ปุ่ม (เมาส์ + คีย์บอร์ด ใช้ action เดียวกัน) ----------
    def _key_to_action(self, key: int) -> Optional[str]:
        ch = key_char(key)
        if ch == "q" or key == 27:
            return "quit"
        enter = key in (13, 10)
        if self.screen == SCREEN_READY:
            if ch in KEY_TO_COLOR:
                return "order:" + KEY_TO_COLOR[ch]
            if ch == "u" or key in (8, 127):  # Backspace บน macOS = 127
                return "undo"
            if enter:
                return "start"
            return {"x": "clear", "d": "dry", "t": "tune", "c": "center"}.get(ch)
        if self.screen == SCREEN_MISSION:
            if ch == "x" or key == 32:
                return "stop"
            return {"s": "fire_toggle", "c": "stop_center"}.get(ch)
        if self.screen == SCREEN_SETUP:
            if enter or key == 32:
                return "next"
            return {"u": "zone_undo", "x": "zone_clear"}.get(ch)
        if self.screen == SCREEN_TUNE:
            if ch in KEY_TO_COLOR:
                return "tab:" + KEY_TO_COLOR[ch]
            if ch in NUDGE_KEYS:
                return "nudge:" + ch
            if enter or key == 32 or ch == "n":
                return "next"
            return {"v": "tab:check", "p": "back", "t": "done", "c": "center"}.get(ch)
        return None

    def _do(self, action: str):
        if action == "quit":
            self.running = False
        elif self.screen == SCREEN_READY:
            if action.startswith("order:"):
                self._toggle_order(action.split(":", 1)[1])
            elif action == "undo":
                self._undo_order()
            elif action == "clear":
                self._clear_order()
            elif action == "start":
                self._start_mission(fire=True)
            elif action == "dry":
                self._start_mission(fire=False)
            elif action == "tune":
                self._enter_tune(calibrating=False)
            elif action == "calibrate":
                self._enter_setup()
            elif action == "center":
                self._recenter("กดปุ่ม RECENTER")
        elif self.screen == SCREEN_MISSION:
            if action == "stop":
                self._abort_mission()
            elif action == "fire_toggle":
                self._toggle_auto_fire()
            elif action == "stop_center":
                self._abort_mission()
                self._recenter("กดปุ่ม STOP + RECENTER")
        elif self.screen == SCREEN_SETUP:
            if action == "zone_undo" and self.blind_zones.user_zones:
                self.blind_zones.user_zones.pop()
            elif action == "zone_clear":
                self.blind_zones.user_zones.clear()
            elif action == "next":
                self._enter_tune(calibrating=True)
            elif action == "cancel":
                self._cancel_calibration()
        elif self.screen == SCREEN_TUNE:
            if action.startswith("tab:"):
                tab = action.split(":", 1)[1]
                self.tune_tab = len(COLOR_ORDER) if tab == "check" else COLOR_ORDER.index(tab)
            elif action == "next":
                if self.tune_tab >= len(COLOR_ORDER):
                    self._finish_tuning()
                else:
                    self.tune_tab += 1
            elif action == "back":
                self.tune_tab = max(0, self.tune_tab - 1)
            elif action == "done":
                self._finish_tuning()
            elif action == "cancel":
                self._cancel_calibration()
            elif action == "reset_color" and self.tune_tab < len(COLOR_ORDER):
                color = COLOR_ORDER[self.tune_tab]
                self.color_ranges[color] = HSVRange(*DEFAULT_HSV[color])
                print(f">> คืนค่าเริ่มต้นสี {color.upper()} [{self.color_ranges[color]}] (กด SAVE เพื่อบันทึก)")
            elif action.startswith("nudge:"):
                pitch, yaw = NUDGE_KEYS[action.split(":", 1)[1]]
                safe_call("gimbal move", self.ep_gimbal.move, pitch=pitch, yaw=yaw, pitch_speed=90, yaw_speed=90)
            elif action == "center":
                self._recenter("กดปุ่ม CENTER")

    # ---------- Loop หลัก ----------
    def _main_loop(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.ui.on_mouse)
        while self.running:
            frame = read_frame(self.ep_camera)
            if frame is None:
                # ไม่มีภาพ = ห้ามขยับตามข้อมูลเก่า (แต่ไม่ขัดจังหวะ moveto ที่กำลังหมุนอยู่)
                if self.mission is not None and self.mission.sub == "AIM":
                    self._stop_gimbal()
            else:
                self._process_frame(frame)
            action = self._key_to_action(cv2.waitKey(1) & 0xFF)
            if action is not None:
                self.ui.actions.append(action)
            while self.ui.actions and self.running:
                self._do(self.ui.actions.popleft())

    def _process_frame(self, frame: np.ndarray):
        now = time.monotonic()
        if self._last_frame_time is not None and now > self._last_frame_time:
            fps = 1.0 / (now - self._last_frame_time)
            self._fps = fps if self._fps == 0 else 0.9 * self._fps + 0.1 * fps
        self._last_frame_time = now

        fd = prepare_frame(frame)
        targets: List[Target] = []
        match, error, mask = None, None, None
        if self.screen == SCREEN_MISSION:
            self.mission.step(fd)
            state, targets = self.mission.state, self.mission.visible
            match, error = self.mission.match, self.mission.error
            if self.mission.done:
                self._finish_mission()
        elif self.screen == SCREEN_SETUP:
            state = f"{len(self.blind_zones.user_zones)} blind zone(s) drawn"
        elif self.screen == SCREEN_TUNE and self.tune_tab < len(COLOR_ORDER):
            color = COLOR_ORDER[self.tune_tab]
            mask = build_mask(fd.hsv, self.color_ranges[color], self.blind_zones)
            targets = find_targets(fd, mask, color)
            state = f"Tuning {color.upper()}: {len(targets)} target(s) found"
        else:  # READY และหน้าตรวจรวมทุกสี
            targets = detect_all_colors(fd, self.color_ranges, self.blind_zones)
            state = f"{len(targets)} target(s) in view"

        self._render(fd, targets, match, error, state, mask)

    # ---------- วาดหน้าจอ ----------
    def _render(self, fd: FrameData, targets, match, error, state, mask):
        canvas = np.full((CANVAS_H, CANVAS_W, 3), UI_BG, np.uint8)
        w, h = fd.size
        if (w, h) == (VIEW_W, VIEW_H):  # สตรีม 540p = ขนาดช่องภาพพอดี ไม่ต้องย่อ
            canvas[:VIEW_H, :VIEW_W] = fd.bgr
        else:
            canvas[:VIEW_H, :VIEW_W] = cv2.resize(fd.bgr, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
        self._draw_view(canvas[:VIEW_H, :VIEW_W], VIEW_W / w, targets, match)
        cv2.rectangle(canvas, (VIEW_W, 0), (CANVAS_W, CANVAS_H), UI_PANEL, -1)
        self.ui.begin()
        self._draw_panel(canvas, state, error)
        self._draw_strip(canvas, fd.bgr, mask, targets)
        cv2.imshow(WINDOW_NAME, canvas)

    def _draw_view(self, view, scale, targets, match):
        """ ภาพกล้อง: พื้นที่ตาบอด, จุดเล็ง, เป้าที่ตรวจเจอ """
        if self.screen == SCREEN_SETUP:
            self.blind_zones.draw(view, fill_alpha=0.55)
            d = self.ui.zone_drawer
            if d is not None and d.drag_start is not None and d.drag_end is not None:
                cv2.rectangle(view, d.drag_start, d.drag_end, (0, 255, 255), 2)
            put_text(view, "Drag on this image to hide anything that is NOT a target", (12, 28), (0, 255, 255), 0.6)
            return

        self.blind_zones.draw(view)
        ax, ay = degrees_to_pixel(AIM_OFFSET_X_DEG, AIM_OFFSET_Y_DEG, VIEW_W, VIEW_H)
        tol_x = max(3, int(LOCK_TOLERANCE_X_DEG * VIEW_W / FOV_X))
        tol_y = max(3, int(LOCK_TOLERANCE_Y_DEG * VIEW_H / FOV_Y))
        cv2.rectangle(view, (ax - tol_x, ay - tol_y), (ax + tol_x, ay + tol_y), (200, 200, 200), 1)
        cv2.drawMarker(view, (ax, ay), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)

        draw_targets(view, targets, scale)
        if match is not None:
            bgr = DISPLAY_COLORS[match.color]
            mx, my = int(match.cx * scale), int(match.cy * scale)
            cv2.circle(view, (mx, my), 12, bgr, 2)
            cv2.line(view, (ax, ay), (mx, my), bgr, 2)
        if time.monotonic() < self._fire_banner_until:
            put_text(view, "FIRE!", (ax - 40, ay - tol_y - 12), (0, 0, 255), 1.0, 3)

    def _screen_badge(self) -> Tuple[str, Tuple[int, int, int]]:
        if self.screen == SCREEN_SETUP:
            return "CALIBRATE 1/2: BLIND ZONES", UI_ORANGE
        if self.screen == SCREEN_TUNE:
            return ("CALIBRATE 2/2: COLORS" if self._calibrating else "TUNE COLORS"), UI_ORANGE
        if self.screen == SCREEN_MISSION:
            return ("MISSION - FIRING", UI_RED) if self.fire_ctrl.armed else ("MISSION - DRY RUN", UI_BLUE)
        return "READY", UI_GREEN

    def _draw_panel(self, canvas, state: str, error):
        """ แผงขวา: สถานะ + ปุ่มของหน้าจอปัจจุบัน """
        px, bw = VIEW_W + 16, PANEL_W - 32
        label(canvas, "ROBOMASTER AUTO-AIM", px, 28, UI_TEXT, 0.6, 2)
        badge, badge_color = self._screen_badge()
        cv2.rectangle(canvas, (px, 40), (px + bw, 70), badge_color, -1)
        label(canvas, badge, px + 10, 61, text_color_for(badge_color), 0.55, 1, max_w=bw - 20)

        fire_text, fire_color = ("FIRE: ARMED", (90, 90, 255)) if self.fire_ctrl.armed else ("FIRE: SAFE", UI_OK)
        label(canvas, f"FPS {self._fps:.0f}", px, 94, UI_DIM)
        label(canvas, fire_text, px + 100, 94, fire_color, 0.5, 2)
        for i, line in enumerate(textwrap.wrap(state, 36)[:2]):
            label(canvas, line, px, 116 + i * 18, UI_TEXT, 0.45)
        if error is not None:
            label(canvas, f"Error  X {error[0]:+.1f}  Y {error[1]:+.1f} deg", px, 152, UI_DIM, 0.45)

        panels = {SCREEN_READY: self._panel_ready, SCREEN_MISSION: self._panel_mission,
                  SCREEN_SETUP: self._panel_setup, SCREEN_TUNE: self._panel_tune}
        panels[self.screen](canvas, px, 172, bw)
        self.ui.button(canvas, px, CANVAS_H - 50, bw, 36, "QUIT (q)", "quit", (70, 60, 110))

    def _color_chips(self, canvas, px, y, bw, action_prefix, is_selected, caption_for):
        chip_w = (bw - 3 * 8) // 4
        for i, color in enumerate(COLOR_ORDER):
            selected = is_selected(color)
            self.ui.button(canvas, px + i * (chip_w + 8), y, chip_w, 42, caption_for(color),
                           action_prefix + color, DISPLAY_COLORS[color] if selected else dim(DISPLAY_COLORS[color]),
                           selected=selected)

    def _panel_ready(self, canvas, px, y, bw):
        half = (bw - 8) // 2
        label(canvas, "FIRE ORDER - click colors in order", px, y, UI_DIM, 0.45, max_w=bw)

        def caption(color):
            short = COLOR_SHORT[color]
            return f"{self.fire_order.index(color) + 1}:{short}" if color in self.fire_order else short

        self._color_chips(canvas, px, y + 10, bw, "order:", lambda c: c in self.fire_order, caption)
        y += 72
        label(canvas, f"Order: {self._order_text(' > ') or '(none)'}", px, y, (0, 255, 255), 0.5, 1, max_w=bw)
        y += 12
        self.ui.button(canvas, px, y, half, 34, "UNDO (u)", "undo")
        self.ui.button(canvas, px + half + 8, y, half, 34, "CLEAR (x)", "clear")
        y += 48
        can_start = bool(self.fire_order)
        self.ui.button(canvas, px, y, bw, 58, "START FIRING (ENTER)", "start", UI_RED, enabled=can_start)
        y += 68
        self.ui.button(canvas, px, y, bw, 38, "DRY RUN - AIM ONLY (d)", "dry", UI_BLUE, enabled=can_start)
        y += 56
        self.ui.button(canvas, px, y, half, 38, "TUNE COLORS (t)", "tune")
        self.ui.button(canvas, px + half + 8, y, half, 38, "CALIBRATE ALL", "calibrate")
        y += 48
        self.ui.button(canvas, px, y, bw, 36, "RECENTER GIMBAL (c)", "center")

    def _panel_mission(self, canvas, px, y, bw):
        self.ui.button(canvas, px, y, bw, 70, "STOP (x / SPACE)", "stop", UI_RED)
        y += 84
        if self.fire_ctrl.armed:
            self.ui.button(canvas, px, y, bw, 44, "FIRE ON - click for SAFE (s)", "fire_toggle", (40, 40, 150))
        else:
            self.ui.button(canvas, px, y, bw, 44, "SAFE - click to ARM FIRE (s)", "fire_toggle", UI_GREEN)
        y += 56
        self.ui.button(canvas, px, y, bw, 38, "STOP + RECENTER (c)", "stop_center")
        y += 60
        label(canvas, f"Order: {self._order_text(' > ')}", px, y, (0, 255, 255), 0.5, 1, max_w=bw)

    def _panel_setup(self, canvas, px, y, bw):
        half = (bw - 8) // 2
        for line in ("Drag on the camera image to", "hide things that are NOT targets.",
                     "Orange box = robot's own gimbal", "(always hidden)."):
            label(canvas, line, px, y, UI_TEXT, 0.48)
            y += 20
        y += 6
        label(canvas, f"Blind zones drawn: {len(self.blind_zones.user_zones)}", px, y, (0, 255, 255))
        y += 14
        self.ui.button(canvas, px, y, half, 34, "UNDO ZONE (u)", "zone_undo")
        self.ui.button(canvas, px + half + 8, y, half, 34, "CLEAR (x)", "zone_clear")
        y += 50
        self.ui.button(canvas, px, y, bw, 56, "NEXT: TUNE COLORS (ENTER)", "next", UI_GREEN)
        y += 70
        self.ui.button(canvas, px, y, bw, 34, "CANCEL (keep saved values)", "cancel")

    def _panel_tune(self, canvas, px, y, bw):
        half = (bw - 8) // 2
        third = (bw - 16) // 3
        label(canvas, "COLOR TO TUNE (r/g/b/y)", px, y, UI_DIM, 0.45)
        tab = self.tune_tab
        self._color_chips(canvas, px, y + 10, bw, "tab:",
                          lambda c: tab < len(COLOR_ORDER) and COLOR_ORDER[tab] == c, lambda c: COLOR_SHORT[c])
        y += 60
        self.ui.button(canvas, px, y, bw, 32, "CHECK ALL COLORS (v)", "tab:check",
                       selected=tab == len(COLOR_ORDER))
        y += 50
        label(canvas, "MOVE GIMBAL (target off-screen?)", px, y, UI_DIM, 0.45, max_w=bw)
        y += 10
        mid = px + third + 8
        self.ui.button(canvas, mid, y, third, 30, "UP (i)", "nudge:i")
        y += 36
        self.ui.button(canvas, px, y, third, 30, "LEFT (j)", "nudge:j")
        self.ui.button(canvas, mid, y, third, 30, "CENTER (c)", "center")
        self.ui.button(canvas, px + 2 * (third + 8), y, third, 30, "RIGHT (l)", "nudge:l")
        y += 36
        self.ui.button(canvas, mid, y, third, 30, "DOWN (k)", "nudge:k")
        y += 46
        last = tab >= len(COLOR_ORDER)
        self.ui.button(canvas, px, y, half, 34, "< BACK (p)", "back", enabled=tab > 0)
        self.ui.button(canvas, px + half + 8, y, half, 34, "FINISH (n)" if last else "NEXT > (n)", "next")
        y += 46
        self.ui.button(canvas, px, y, bw, 44, "SAVE & FINISH (t)", "done", UI_GREEN)
        y += 54
        self.ui.button(canvas, px, y, bw, 32, "CANCEL (keep saved values)", "cancel")
        y += 42
        self.ui.button(canvas, px, y, bw, 32, "RESET THIS COLOR TO DEFAULT", "reset_color",
                       enabled=tab < len(COLOR_ORDER))

    def _draw_strip(self, canvas, frame, mask, targets: List[Target]):
        """ แถบล่าง: slider จูนสี + ภาพ Mask หรือคำแนะนำ/ผลภารกิจ """
        y0 = VIEW_H
        cv2.line(canvas, (0, y0), (VIEW_W, y0), (80, 80, 80), 1)
        if self.screen == SCREEN_TUNE and self.tune_tab < len(COLOR_ORDER):
            self._strip_sliders(canvas, frame, mask, COLOR_ORDER[self.tune_tab])
            return

        counts = " | ".join(f"{c.upper()} {sum(t.color == c for t in targets)}" for c in COLOR_ORDER)
        if self.screen == SCREEN_SETUP:
            lines = [("STEP 1/2 - BLIND ZONES", UI_ORANGE),
                     ("Mouse: press, drag and release on the camera image = add one blind zone", UI_TEXT),
                     ("The robot will never see or shoot inside the dark boxes.", UI_TEXT)]
            keys = "u = undo zone   x = clear zones   ENTER = next step   q = quit"
        elif self.screen == SCREEN_TUNE:
            lines = [("CHECK ALL COLORS", UI_ORANGE), (f"In view: {counts}", (0, 255, 255)),
                     ("Every target should show exactly ONE correct label.", UI_TEXT),
                     ("Wrong or double label? Click that color above and tune it again.", UI_TEXT)]
            keys = "r/g/b/y = tune color   p = back   t = save & finish   q = quit"
        elif self.screen == SCREEN_MISSION:
            m = self.mission
            lines = [(m.state, UI_TEXT)] + [(text, (0, 255, 255)) for text in m.progress_lines(per_line=8)]
            keys = "x / SPACE = stop now   s = fire on/off   c = stop + recenter   q = quit"
        else:
            lines = [(f"In view: {counts}", (0, 255, 255)),
                     ("Click color chips (or r/g/b/y) in firing order, then START FIRING.", UI_TEXT),
                     ("The robot scans around first, then fires at every target of those colors.", UI_TEXT)]
            lines += [(text, UI_DIM) for text in self._last_result]
            keys = "r/g/b/y = order  u = undo  x = clear  ENTER = fire  d = dry run  t = tune  c = center  q = quit"

        for i, (text, color) in enumerate(lines[:6]):
            label(canvas, text, 16, y0 + 26 + i * 22, color, 0.5, 1, max_w=VIEW_W - 32)
        label(canvas, keys, 16, CANVAS_H - 12, UI_DIM, 0.45, 1, max_w=VIEW_W - 32)

    def _strip_sliders(self, canvas, frame, mask, color: str):
        y0 = VIEW_H
        rng = self.color_ranges[color]
        sliders = (("H min", "h_min", 179, True), ("H max", "h_max", 179, True),
                   ("S min", "s_min", 255, False), ("S max", "s_max", 255, False),
                   ("V min", "v_min", 255, False), ("V max", "v_max", 255, False))
        for i, (text, name, max_value, hue) in enumerate(sliders):
            x = 20 + (i % 2) * 310
            y = y0 + 12 + (i // 2) * 50
            self.ui.slider(canvas, x, y, 290, text, getattr(rng, name), max_value,
                           self._slider_setter(color, name), hue)
        hint = "H min > H max = hue wraps around (used for RED)" if rng.wraps else \
            "S min up = ignore pale colors   V min up = ignore shadows"
        label(canvas, hint, 20, CANVAS_H - 12, UI_DIM, 0.45)

        # ภาพเฉพาะพิกเซลที่ผ่านช่วงสี (สีจริง) ด้านขวาของแถบล่าง
        preview = cv2.bitwise_and(frame, frame, mask=mask)
        canvas[y0:CANVAS_H, VIEW_W - 320:VIEW_W] = cv2.resize(preview, (320, STRIP_H), interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (VIEW_W - 320, y0), (VIEW_W - 1, CANVAS_H - 1), (90, 90, 90), 1)
        put_text(canvas, f"MASK PREVIEW: {color.upper()}", (VIEW_W - 312, y0 + 20), (255, 255, 255), 0.5, 1)

    def _slider_setter(self, color: str, name: str) -> Callable[[int], None]:
        def setter(value: int):
            self.color_ranges[color] = replace(self.color_ranges[color], **{name: value})
        return setter

    # ---------- อื่นๆ ----------
    def _update_status_led(self):
        self.leds.show_status("white", self.fire_ctrl.armed, self.is_tuning)

    def _save(self):
        save_config(CONFIG_PATH, self.color_ranges, self.blind_zones.user_zones, self.fire_order)


