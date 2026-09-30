"""aim: mission."""

from .helpers import pixel_to_degrees, safe_call
from .settings import (
    AIM_OFFSET_X_DEG,
    AIM_OFFSET_Y_DEG,
    CROSS_POSE_MERGE_DEG,
    DRY_RUN_HOLD_S,
    ENGAGE_TIMEOUT_S,
    LOST_REPOINT_S,
    MATCH_AREA_RATIO,
    MATCH_RADIUS_DEG,
    MOVE_SETTLE_S,
    MOVE_SPEED,
    MOVE_TIMEOUT_S,
    PITCH_LIMITS,
    POSE_MERGE_DEG,
    SCAN_FRAMES,
    SCAN_MIN_HITS,
    SCAN_PITCH_ANGLES,
    SCAN_YAW_ANGLES,
    SHOTS_PER_TARGET,
)
from .vision import FrameData, Target, build_mask, detect_all_colors, find_targets
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
import math
import time

# ==========================================
# 10. ภารกิจยิงทุกเป้าทุกสี
# ==========================================
@dataclass
class PlannedTarget:
    """ เป้าหนึ่งเป้า ตำแหน่งเป็นมุมเทียบกับหน้าหุ่น (yaw + = ขวา, pitch + = บน) """
    color: str
    yaw: float
    pitch: float
    area: float
    offset: float                                  # ระยะจากกึ่งกลางภาพตอนวัด (น้อย = แม่น ขอบเลนส์บิดเบี้ยว)
    hits: int = 1
    poses: Set[int] = field(default_factory=set)   # มุมกวาดที่เห็นเป้านี้
    status: str = "PENDING"
    shots: int = 0


def angular_distance(a: PlannedTarget, b: PlannedTarget) -> float:
    return math.hypot(a.yaw - b.yaw, a.pitch - b.pitch)


class TargetMap:
    """ รวมผลการกวาดหาเป้าจากหลายมุมกล้อง ให้เป็นรายการเป้าที่ไม่ซ้ำกัน

    - ในมุมกวาดเดียวกัน: หลายเฟรมที่เห็นจุดเดียวกัน (< POSE_MERGE_DEG) = เป้าเดียว
    - ข้ามมุมกวาด: ค่ามุมเพี้ยนได้หลายองศาเพราะเลนส์มุมกว้าง จึงรวมในรัศมี CROSS_POSE_MERGE_DEG
      แต่ถ้ามุมกวาดเดียวกันเห็นเป็น 2 ชิ้นแยกกัน ถือว่าเป็นคนละเป้าเสมอ
    """

    def __init__(self):
        self._current: List[PlannedTarget] = []
        self._by_pose: List[List[PlannedTarget]] = []

    def add_sighting(self, sighting: PlannedTarget):
        for obs in self._current:
            if obs.color == sighting.color and angular_distance(obs, sighting) < POSE_MERGE_DEG:
                obs.hits += 1
                return
        self._current.append(sighting)

    def end_pose(self) -> int:
        """ ปิดมุมกวาดปัจจุบัน คืนจำนวนเป้าที่เจอในมุมนี้ """
        seen = [o for o in self._current if o.hits >= SCAN_MIN_HITS]
        kept: List[PlannedTarget] = []
        # วัตถุชิ้นเดียวอาจผ่านช่วงสีได้ 2 สี (เช่นส้มผ่านทั้งแดงและเหลือง) เก็บสีที่เห็นบ่อยกว่า
        for obs in sorted(seen, key=lambda o: (o.hits, o.area), reverse=True):
            if all(angular_distance(obs, k) >= POSE_MERGE_DEG for k in kept):
                kept.append(obs)
        self._by_pose.append(kept)
        self._current = []
        return len(kept)

    def build_plan(self, order: List[str]) -> List[PlannedTarget]:
        """ order = สีที่จะยิงเรียงตามลำดับ สีที่ไม่อยู่ใน order จะไม่ถูกยิง """
        observations = sorted(((i, obs) for i, objs in enumerate(self._by_pose) for obs in objs),
                              key=lambda item: item[1].offset)  # ค่าที่วัดใกล้กลางภาพที่สุดเป็นตัวแทน
        clusters: List[PlannedTarget] = []
        for pose_idx, obs in observations:
            candidates = [c for c in clusters if c.color == obs.color and pose_idx not in c.poses
                          and angular_distance(c, obs) < CROSS_POSE_MERGE_DEG]
            if candidates:
                nearest = min(candidates, key=lambda c: angular_distance(c, obs))
                nearest.poses.add(pose_idx)
                nearest.hits += obs.hits
            else:
                obs.poses = {pose_idx}
                clusters.append(obs)
        # กวาดหาทุกสีแล้วค่อยกรอง: วัตถุที่ผ่าน 2 ช่วงสีถูกตัดสินเป็นสีที่เห็นบ่อยกว่าไปแล้วใน end_pose()
        # ลำดับยิง: ตามลำดับสีที่เลือก แล้วซ้ายไปขวา (บนลงล่างถ้า yaw เท่ากัน)
        chosen = [c for c in clusters if c.color in order]
        return sorted(chosen, key=lambda c: (order.index(c.color), c.yaw, -c.pitch))


class Mission:
    """ ภารกิจยิงทุกเป้าตามลำดับสี แบบ non-blocking (เรียก step() ทุกเฟรม ภาพและปุ่มจึงไม่ค้าง)

    SCAN   : หมุนไปทีละมุมใน SCAN_PITCH_ANGLES x SCAN_YAW_ANGLES เก็บตำแหน่งเป้าทุกสี
    ENGAGE : ยิงตามลำดับสีใน order: หมุนไปยังเป้า (moveto) -> เล็งละเอียดด้วยภาพ + PID -> ยิง -> เป้าถัดไป
    DONE   : จบภารกิจ

    ใช้ gimbal.moveto ซึ่งอ้างอิง yaw กับฐานล้อ และ pitch กับแนวระนาบ จึงเป็นพิกัดเดียวกับ recenter
    """
    STATUS_SHORT = {"PENDING": "..", "ENGAGING": ">>", "FIRED": "SHOT", "AIMED-SAFE": "AIM", "SKIPPED": "SKIP"}
    MOVE_RETRY_S = 0.5

    def __init__(self, app: "AutoAimApp", order: List[str]):
        self.app = app
        self.order = list(order)
        self.poses = [(float(p), float(y)) for p in SCAN_PITCH_ANGLES for y in SCAN_YAW_ANGLES]
        self.target_map = TargetMap()
        self.plan: List[PlannedTarget] = []
        self.phase = "SCAN"
        self.sub = "MOVE"  # MOVE -> SETTLE -> COLLECT (SCAN) หรือ AIM (ENGAGE)
        self.index = 0     # มุมกวาดปัจจุบัน (SCAN) / เป้าปัจจุบัน (ENGAGE)
        self.pose = (0.0, 0.0)

        # ผลของเฟรมล่าสุด ให้ App วาด HUD
        self.state = ""
        self.visible: List[Target] = []
        self.match: Optional[Target] = None
        self.error: Optional[Tuple[float, float]] = None

        self._action = None
        self._sub_t0 = 0.0
        self._last_move_try = float("-inf")
        self._collected = 0
        self._target_t0 = 0.0
        self._last_match_deg: Optional[Tuple[float, float]] = None
        self._lost_since: Optional[float] = None
        self._hold_since: Optional[float] = None
        self._move_to(*self.poses[0])

    @property
    def done(self) -> bool:
        return self.phase == "DONE"

    # ---------- การหมุนไปยังตำแหน่ง ----------
    def _move_to(self, pitch: float, yaw: float):
        self.pose = (min(max(pitch, PITCH_LIMITS[0]), PITCH_LIMITS[1]), yaw)
        self.sub = "MOVE"
        self._sub_t0 = time.monotonic()
        self._last_match_deg = None
        safe_call("gimbal stop", self.app.ep_gimbal.drive_speed, pitch_speed=0, yaw_speed=0)
        self._send_move()

    def _send_move(self):
        self._last_move_try = time.monotonic()
        self._action = safe_call("gimbal moveto", self.app.ep_gimbal.moveto, pitch=self.pose[0], yaw=self.pose[1],
                                 pitch_speed=MOVE_SPEED, yaw_speed=MOVE_SPEED)

    def _ready(self) -> bool:
        """ อัปเดตขั้น MOVE -> SETTLE คืน True เมื่อหมุนถึงและรอภาพตามทันครบแล้ว """
        now = time.monotonic()
        if self.sub == "MOVE":
            if self._action is None and now - self._last_move_try >= self.MOVE_RETRY_S:
                self._send_move()  # คำสั่งก่อนหน้าโดนปฏิเสธ (เช่น action เดิมยังไม่จบ) ลองใหม่
            arrived = self._action is not None and self._action.is_completed
            if arrived or now - self._sub_t0 > MOVE_TIMEOUT_S:
                if not arrived:
                    print("[MISSION] ⚠️ Gimbal หมุนไม่เสร็จตามเวลา ใช้ตำแหน่งปัจจุบันต่อ")
                self.sub, self._sub_t0 = "SETTLE", now
            return False
        if self.sub == "SETTLE":
            if now - self._sub_t0 < MOVE_SETTLE_S:
                return False
            self.sub = "COLLECT" if self.phase == "SCAN" else "AIM"
        return True

    # ---------- วนทุกเฟรม ----------
    def step(self, fd: FrameData):
        self.visible, self.match, self.error = [], None, None
        if self.phase == "SCAN":
            self._step_scan(fd)
        elif self.phase == "ENGAGE":
            self._step_engage(fd)

    def _step_scan(self, fd: FrameData):
        pitch, yaw = self.pose
        ready = self._ready()
        self.state = f"SCAN {self.index + 1}/{len(self.poses)} (pitch {pitch:+.0f}, yaw {yaw:+.0f}) {self.sub}"
        if not ready:
            return

        w, h = fd.size
        self.visible = detect_all_colors(fd, self.app.color_ranges, self.app.blind_zones)
        for t in self.visible:
            dx, dy = pixel_to_degrees(t.cx, t.cy, w, h)
            self.target_map.add_sighting(
                PlannedTarget(t.color, yaw=yaw + dx, pitch=pitch + dy, area=t.area, offset=math.hypot(dx, dy)))
        self._collected += 1
        if self._collected < SCAN_FRAMES:
            return

        found = self.target_map.end_pose()
        print(f"[MISSION] 🔍 มุม {self.index + 1}/{len(self.poses)} (pitch {pitch:+.0f}°, yaw {yaw:+.0f}°): "
              f"เห็น {found} เป้า")
        self._collected = 0
        self.index += 1
        if self.index < len(self.poses):
            self._move_to(*self.poses[self.index])
            return

        self.plan = self.target_map.build_plan(self.order)
        if not self.plan:
            print("[MISSION] ไม่พบเป้าสีที่เลือก")
            self.phase = "DONE"
            return
        print(f"[MISSION] 📋 แผนยิง {len(self.plan)} เป้า:")
        for i, t in enumerate(self.plan, 1):
            print(f"   {i}. {t.color.upper():<6} yaw {t.yaw:+6.1f}°  pitch {t.pitch:+5.1f}°")
        self.phase = "ENGAGE"
        self.index = 0
        self._begin_target()

    def _begin_target(self):
        t = self.plan[self.index]
        t.status = "ENGAGING"
        self._target_t0 = time.monotonic()
        self._lost_since = None
        self._hold_since = None
        self.app._reset_aim()
        print(f"[MISSION] 🎯 เป้า {self.index + 1}/{len(self.plan)}: {t.color.upper()}")
        # หันให้เป้าไปตกที่จุดเล็ง (ไม่ใช่กลางภาพ) จึงลบ AIM_OFFSET
        self._move_to(t.pitch - AIM_OFFSET_Y_DEG, t.yaw - AIM_OFFSET_X_DEG)

    def _finish_target(self, status: str):
        self.plan[self.index].status = status
        print(f"[MISSION]    -> {status}")
        self.app._stop_gimbal()
        self.index += 1
        if self.index < len(self.plan):
            self._begin_target()
        else:
            self.phase = "DONE"

    def _step_engage(self, fd: FrameData):
        app = self.app
        t = self.plan[self.index]
        now = time.monotonic()
        label = f"TARGET {self.index + 1}/{len(self.plan)} {t.color.upper()}"
        if now - self._target_t0 > ENGAGE_TIMEOUT_S:
            self._finish_target("SKIPPED")
            return
        if not self._ready():
            self.state = f"{label} {self.sub}"
            return

        # หาเป้าสีนี้ที่ใกล้จุดอ้างอิงที่สุด: เฟรมแรกใช้จุดเล็ง เฟรมต่อไปใช้ตำแหน่งที่เจอล่าสุด (ไม่กระโดดไปเป้าข้างๆ)
        # และขนาดต้องใกล้กับตอนกวาดหา (แบบโค้ดเพื่อน) กันสลับไปเล็งป้ายสีเดียวกันที่อยู่ติดกัน
        w, h = fd.size
        mask = build_mask(fd.hsv, app.color_ranges[t.color], app.blind_zones)
        self.visible = find_targets(fd, mask, t.color)
        ref = self._last_match_deg or (AIM_OFFSET_X_DEG, AIM_OFFSET_Y_DEG)
        best, best_deg, best_dist = None, None, MATCH_RADIUS_DEG
        for cand in self.visible:
            if not MATCH_AREA_RATIO[0] < cand.area / t.area < MATCH_AREA_RATIO[1]:
                continue
            deg = pixel_to_degrees(cand.cx, cand.cy, w, h)
            dist = math.hypot(deg[0] - ref[0], deg[1] - ref[1])
            if dist < best_dist:
                best, best_deg, best_dist = cand, deg, dist

        if best is None:
            app._stop_gimbal()
            app.fire_ctrl.reset_lock()
            app.pid_yaw.reset()
            app.pid_pitch.reset()
            self._last_match_deg = None
            self._hold_since = None
            if self._lost_since is None:
                self._lost_since = now
            elif now - self._lost_since > LOST_REPOINT_S:
                self._lost_since = None
                self._move_to(t.pitch - AIM_OFFSET_Y_DEG, t.yaw - AIM_OFFSET_X_DEG)
            self.state = f"{label} LOST"
            return

        self._lost_since = None
        self.match, self._last_match_deg = best, best_deg
        err_x, err_y = best_deg[0] - AIM_OFFSET_X_DEG, best_deg[1] - AIM_OFFSET_Y_DEG
        self.error = (err_x, err_y)
        safe_call("gimbal drive", app.ep_gimbal.drive_speed,
                  pitch_speed=app.pid_pitch.compute(err_y), yaw_speed=app.pid_yaw.compute(err_x))

        fire_state, should_fire = app.fire_ctrl.evaluate(err_x, err_y, app.is_tuning)
        self.state = f"{label} {fire_state}"
        if should_fire:
            if app._fire():
                t.shots += 1
            if t.shots >= SHOTS_PER_TARGET:
                self._finish_target("FIRED")
            return
        if app.fire_ctrl.settled and not app.fire_ctrl.armed:
            # Auto-Fire ปิด = ซ้อมเล็ง: ค้างให้ดูสักครู่แล้วไปเป้าถัดไป
            if self._hold_since is None:
                self._hold_since = now
            elif now - self._hold_since >= DRY_RUN_HOLD_S:
                self._finish_target("AIMED-SAFE")
        else:
            self._hold_since = None

    # ---------- สรุปผล ----------
    def progress_lines(self, per_line: int = 6) -> List[str]:
        tokens = [f"{i}.{t.color[0].upper()}:{self.STATUS_SHORT[t.status]}" for i, t in enumerate(self.plan, 1)]
        return ["  ".join(tokens[i:i + per_line]) for i in range(0, len(tokens), per_line)]

    def print_summary(self, aborted: bool = False):
        print("\n=== สรุปภารกิจ" + (" (ถูกยกเลิก)" if aborted else "") + " ===")
        if not self.plan:
            print("  ยังกวาดหาเป้าไม่เสร็จ" if self.phase == "SCAN" else "  ไม่พบเป้าสีที่เลือก")
            return
        for i, t in enumerate(self.plan, 1):
            print(f"  {i}. {t.color.upper():<6} yaw {t.yaw:+6.1f}°  pitch {t.pitch:+5.1f}°  -> {t.status}")
        fired = sum(t.status == "FIRED" for t in self.plan)
        print(f"  ยิงแล้ว {fired}/{len(self.plan)} เป้า\n")


