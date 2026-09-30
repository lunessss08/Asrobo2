"""maze: ui."""

from .camera import FrameGrabber
from .mission import MazeMission
from .settings import (
    CLEAR_CONFIRM_S,
    CLEAR_MIN_GAP_S,
    Cell,
    DIRS,
    DIR_NAMES,
    END_MODES,
    ORDER_MODES,
    SHAPE_SHORT,
    START_MODES,
    WINDOW,
)
from robo_app import aim as gs
from robo_app.aim.widgets import Button
from typing import List
from typing import Optional
from typing import Tuple
import cv2
import numpy as np
import time

# ==========================================
# 6. GUI หน้าต่างเดียว: ภาพกล้อง + แผนที่สด + ปุ่ม (วิดเจ็ตจาก gimbalshoot)
# ==========================================
CANVAS_W, CANVAS_H = 1280, 720
CAM_W, CAM_H = 640, 360            # ภาพกล้อง มุมซ้ายบน
MAP_Y0 = CAM_H                     # แผนที่ อยู่ใต้ภาพกล้อง (640 x 360)
PANEL_X = 640                      # แผงปุ่ม ครึ่งขวา
BG = (24, 23, 27)
PANEL = (34, 33, 39)
CARD = (43, 42, 49)
BUTTON = (65, 63, 72)
TEXT = (240, 239, 239)
MUTED = (168, 163, 166)
ACCENT = (216, 198, 70)
GREEN = (145, 209, 101)
RED = (77, 72, 207)
BLUE = (175, 126, 77)
WALL_COLOR = (97, 207, 231)
ROUTE_COLOR = ACCENT


def rounded_rect(canvas, x, y, w, h, color, radius=10):
    radius = min(radius, w // 2, h // 2)
    cv2.rectangle(canvas, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(canvas, (x, y + radius), (x + w, y + h - radius), color, -1)
    for cx, cy in ((x + radius, y + radius), (x + w - radius, y + radius),
                   (x + radius, y + h - radius), (x + w - radius, y + h - radius)):
        cv2.circle(canvas, (cx, cy), radius, color, -1)


class App:
    def __init__(self, mission: MazeMission, grabber: FrameGrabber):
        self.m, self.grabber = mission, grabber
        self.ui = gs.Ui()
        self.start_mode = self.end_mode = self.order_mode = 0
        self.color_choices = (None, "red", "green", "blue", "yellow")
        self.shape_choices = (None, "circle", "square", "rect")
        self.color_filter = self.color_choices.index("red")
        self.shape_filter = self.shape_choices.index("circle")
        self.show_options = False
        self.picks: List[Cell] = []   # ลำดับเป้าที่คลิกบนแผนที่ (PICK ORDER)
        self.goal: Optional[Cell] = None
        self.running = True
        self._grid = (0, 0, 1, 0, 0, 1, 1)  # (x0, y0, ขนาดช่อง px, ช่องซ้ายสุด, ช่องล่างสุด, กว้าง, สูง) ที่วาดล่าสุด
        self._clear_until = 0.0  # กด n แล้ว ต้องกด y ก่อนเวลานี้ถึงจะล้างจริง (กดค้างปุ่มเดียวยืนยันเองไม่ได้)
        self._clear_armed = 0.0  # เวลาที่กด n (ยืนยันเร็วเกิน CLEAR_MIN_GAP_S ไม่นับ กันดับเบิลคลิก)

    # ---------- เมาส์ + คีย์บอร์ด ----------
    def on_mouse(self, event, x, y, flags, param):
        x0, y0, cs, wx, wy, ww, wh = self._grid
        maze = self.m.maze
        if event == cv2.EVENT_LBUTTONDOWN and x0 <= x < x0 + ww * cs and y0 <= y < y0 + wh * cs:
            cx, cy = wx + (x - x0) // cs, wy + wh - 1 - (y - y0) // cs
            self.ui.actions.append(f"{'pick' if (cx, cy) in maze.targets else 'goal'}:{cx},{cy}")
            return
        self.ui.on_mouse(event, x, y, flags, param)

    def key_action(self, key: int) -> Optional[str]:
        ch = gs.key_char(key)
        if ch == "q" or key == 27:
            return "quit"
        if ch == "x" or key == 32:
            return "stop"
        if key in (13, 10):
            return "shoot"
        return {"e": "explore", "f": "finish", "d": "dry", "c": "center", "h": "home", "n": "clear",
                "y": "confirm_clear", "1": "opt_start", "2": "opt_end", "3": "opt_order", "4": "filter_color", "5": "filter_shape", "m": "map_only", "u": "undo"}.get(ch)

    def do(self, action: str):
        m = self.m
        if action == "quit":
            m.stop()
            self.running = False
        elif action == "stop":
            m.stop()
        elif action == "options":
            self.show_options = not self.show_options
        elif m.busy and action != "finish":
            return
        elif m.pose_uncertain and action in ("explore", "map_only", "shoot", "dry"):
            m.log("Place robot at START, then CONFIRM START (h) before running")
            return
        elif action.startswith("goal:"):
            self.goal = tuple(int(v) for v in action[5:].split(","))
            m.log(f"GOAL set to {self.goal}")
        elif action.startswith("pick:"):
            cell = tuple(int(v) for v in action[5:].split(","))
            if cell in self.picks:
                self.picks.remove(cell)
            else:
                self.picks.append(cell)
            self.sync_order_mode()
            m.log("PICK: " + (" > ".join(f"{m.maze.target_label(c)}{c}" for c in self.picks) or "empty -> NEAREST"))
        elif action == "opt_start":
            self.start_mode = (self.start_mode + 1) % len(START_MODES)
        elif action == "opt_end":
            self.end_mode = (self.end_mode + 1) % len(END_MODES)
        elif action == "opt_order":
            self.order_mode = (self.order_mode + 1) % len(ORDER_MODES)
        elif action == "undo":
            if self.picks:
                self.picks.pop()
            self.sync_order_mode()
        elif action == "clear_picks":
            self.picks = []
            self.sync_order_mode()
        elif action == "finish":
            if m.busy and m.phase == "EXPLORE":
                m.finish_requested = True
                m.log("Finishing round 1 at the next cell centre...")
            elif not m.busy and not m.maze.explored:
                m.finish_round1()
        elif action.startswith("color:"):
            value = action.split(":", 1)[1]
            if value in ("any", "red", "green", "blue", "yellow"):
                self.color_filter = self.color_choices.index(None if value == "any" else value)
        elif action.startswith("shape:"):
            value = action.split(":", 1)[1]
            if value in ("any", "circle", "square", "rect"):
                self.shape_filter = self.shape_choices.index(None if value == "any" else value)
        elif action == "filter_color":
            self.color_filter = (self.color_filter + 1) % len(self.color_choices)
        elif action == "filter_shape":
            self.shape_filter = (self.shape_filter + 1) % len(self.shape_choices)
        elif action == "explore":
            color, shape = self.color_choices[self.color_filter], self.shape_choices[self.shape_filter]
            if color is None and shape is None:
                m.log("Choose COLOR (4) or SHAPE (5) before EXPLORE + FIRE")
                return
            m.start_job("EXPLORE", m.explore, True, {color} if color else None, {shape} if shape else None)
        elif action == "map_only":
            m.start_job("EXPLORE", m.explore, False)
        elif action in ("shoot", "dry"):
            if not m.maze.explored:
                m.log("Run EXPLORE (round 1) first")
                return
            color, shape = self.color_choices[self.color_filter], self.shape_choices[self.shape_filter]
            m.start_job("SHOOT" if action == "shoot" else "DRY RUN", m.shoot_run, action == "shoot",
                        START_MODES[self.start_mode], END_MODES[self.end_mode], ORDER_MODES[self.order_mode],
                        list(self.picks), self.goal, {color} if color else None, {shape} if shape else None)
        elif action == "center":
            m.start_job("RECENTER", m.recenter)
        elif action == "home":
            m.start_job("AT START", m.reset_pose)
        elif action == "clear":
            self._clear_armed = time.monotonic()
            self._clear_until = self._clear_armed + CLEAR_CONFIRM_S
            m.log(f"Erase the whole map? Press y (or click SURE?) within {CLEAR_CONFIRM_S:.0f} s")
        elif action == "confirm_clear":
            now = time.monotonic()
            if self._clear_armed + CLEAR_MIN_GAP_S <= now < self._clear_until:
                self._clear_until = 0.0
                self.picks, self.goal = [], None
                m.start_job("NEW MAP", m.clear_map)

    def sync_order_mode(self):
        """ มีเป้าที่คลิกไว้ = PICK ORDER / ไม่มีเลย = NEAREST (กันยิงรอบ 2 ด้วยรายการว่าง) """
        self.order_mode = ORDER_MODES.index("PICK ORDER" if self.picks else "NEAREST")

    # ---------- Loop หลัก ----------
    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while self.running:
            self.render()
            action = self.key_action(cv2.waitKey(30) & 0xFF)
            if action is not None:
                self.ui.actions.append(action)
            while self.ui.actions and self.running:
                self.do(self.ui.actions.popleft())

    def button(self, canvas, x, y, w, h, text, action, color=BUTTON, enabled=True, selected=False):
        """Maze-only button skin; keep the shared widget's hit testing/actions."""
        button = Button(x, y, w, h, text, action, color, enabled, selected)
        self.ui.buttons.append(button)
        fill = color if enabled else (52, 50, 56)
        if enabled and button.contains(*self.ui.mouse):
            fill = tuple(min(255, channel + 17) for channel in fill)
        rounded_rect(canvas, x, y, w, h, fill, radius=8)
        if selected:
            cv2.rectangle(canvas, (x + 2, y + 2), (x + w - 2, y + 3), TEXT, -1)
        scale = 0.52 if h >= 40 else 0.46
        while scale > 0.35 and cv2.getTextSize(text, gs.FONT, scale, 1)[0][0] > w - 18:
            scale -= 0.02
        (tw, th), _ = cv2.getTextSize(text, gs.FONT, scale, 1)
        foreground = gs.text_color_for(fill) if enabled else (111, 108, 115)
        cv2.putText(canvas, text, (x + (w - tw) // 2, y + (h + th) // 2),
                    gs.FONT, scale, foreground, 1, cv2.LINE_AA)

    # ---------- วาดหน้าจอ ----------
    def render(self):
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, np.uint8)
        self.draw_camera(canvas)
        self.draw_map(canvas)
        cv2.rectangle(canvas, (PANEL_X, 0), (CANVAS_W, CANVAS_H), PANEL, -1)
        cv2.line(canvas, (PANEL_X, 0), (PANEL_X, CANVAS_H), (58, 55, 59), 1)
        self.ui.begin()
        self.draw_panel(canvas)
        cv2.imshow(WINDOW, canvas)

    def draw_camera(self, canvas):
        frame = self.grabber.frame
        if frame is None:
            cv2.circle(canvas, (CAM_W // 2, CAM_H // 2 - 14), 28, (83, 79, 84), 2)
            cv2.circle(canvas, (CAM_W // 2, CAM_H // 2 - 14), 12, (83, 79, 84), 2)
            gs.label(canvas, "WAITING FOR CAMERA", CAM_W // 2 - 95, CAM_H // 2 + 44, MUTED, 0.55, 1)
        else:
            view = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
            gs.draw_targets(view, list(self.m.overlay), CAM_W / max(1, self.m.overlay_w))
            ax, ay = gs.degrees_to_pixel(gs.AIM_OFFSET_X_DEG, gs.AIM_OFFSET_Y_DEG, CAM_W, CAM_H)
            cv2.drawMarker(view, (ax, ay), TEXT, markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)
            canvas[:CAM_H, :CAM_W] = view
        cv2.rectangle(canvas, (0, 0), (CAM_W, 37), (24, 23, 27), -1)
        cv2.circle(canvas, (19, 19), 5, GREEN if frame is not None else MUTED, -1)
        gs.label(canvas, "LIVE CAMERA" if frame is not None else "CAMERA OFFLINE", 32, 24, TEXT, 0.52, 2)
        gs.label(canvas, f"CELL {self.m.cell[0]},{self.m.cell[1]}  /  {DIR_NAMES[self.m.heading]}",
                 CAM_W - 148, 24, MUTED, 0.48, 1)
        cv2.rectangle(canvas, (0, CAM_H - 38), (CAM_W, CAM_H), (24, 23, 27), -1)
        slam_info = f"SLAM {self.m.slam.confidence:.0%}  |  " if self.m.slam is not None else ""
        gs.label(canvas, (slam_info + self.m.status)[:72], 16, CAM_H - 14, TEXT, 0.49, 1, max_w=CAM_W - 32)

    def map_window(self) -> Tuple[int, int, int, int]:
        """ ช่วงช่องที่จะวาด (x, y ล่างซ้าย, กว้าง, สูง): ตารางเผื่อ 13x13 วาดเฉพาะส่วนที่รู้แล้ว + ขอบ 1 ช่อง
        ช่องจะใหญ่พอให้อ่านสี/รูปทรงเป้าและคลิกเลือกได้ """
        m, maze = self.m, self.m.maze
        if maze.field is None:
            return 0, 0, maze.w, maze.h
        cells = set(maze.visited) | set(maze.targets) | {m.start_cell, m.cell}
        xs, ys = [x for x, _ in cells], [y for _, y in cells]
        x1, x2 = max(0, min(xs) - 1), min(maze.w - 1, max(xs) + 1)
        y1, y2 = max(0, min(ys) - 1), min(maze.h - 1, max(ys) + 1)
        return x1, y1, x2 - x1 + 1, y2 - y1 + 1

    def draw_map(self, canvas):
        m, maze = self.m, self.m.maze
        gs.label(canvas, "FIELD MAP", 20, MAP_Y0 + 28, TEXT, 0.56, 2)
        gs.label(canvas, f"{maze.w} x {maze.h} CELLS", CAM_W - 124, MAP_Y0 + 28, MUTED, 0.43, 1)
        cv2.line(canvas, (20, MAP_Y0 + 38), (CAM_W - 20, MAP_Y0 + 38), (54, 52, 58), 1)
        wx, wy, ww, wh = self.map_window()
        cs = int(min((CAM_W - 170) / ww, (CANVAS_H - MAP_Y0 - 80) / wh, 70))
        x0, y0 = 20, MAP_Y0 + 53 + ((CANVAS_H - MAP_Y0 - 80) - wh * cs) // 2
        self._grid = (x0, y0, cs, wx, wy, ww, wh)

        def inside(c: Cell) -> bool:
            return wx <= c[0] < wx + ww and wy <= c[1] < wy + wh

        def corner(c: Cell) -> Tuple[int, int]:
            return x0 + (c[0] - wx) * cs, y0 + (wy + wh - 1 - c[1]) * cs

        def center(c: Cell) -> Tuple[int, int]:
            px, py = corner(c)
            return px + cs // 2, py + cs // 2

        targets, visited, results = dict(maze.targets), set(maze.visited), dict(m.results)
        for x in range(wx, wx + ww):
            for y in range(wy, wy + wh):
                c = (x, y)
                px, py = corner(c)
                if c in targets:
                    fill = gs.dim(gs.DISPLAY_COLORS[targets[c]], 0.46)
                elif c in visited:
                    fill = (64, 78, 60)
                elif maze.explored:
                    fill = (35, 34, 39)  # สำรวจครบแล้วไปไม่ถึง = บล็อก/ห้องปิด
                else:
                    fill = (47, 46, 53)
                cv2.rectangle(canvas, (px + 1, py + 1), (px + cs - 1, py + cs - 1), fill, -1)
                cv2.rectangle(canvas, (px, py), (px + cs, py + cs), (76, 73, 78), 1)
                if maze.explored and c not in visited and c not in targets:
                    cv2.line(canvas, (px + 6, py + 6), (px + cs - 6, py + cs - 6), (70, 70, 70), 1)
                    cv2.line(canvas, (px + cs - 6, py + 6), (px + 6, py + cs - 6), (70, 70, 70), 1)
                if c in targets:
                    shape = SHAPE_SHORT.get(maze.target_shape.get(c, "?"), "?")
                    gs.label(canvas, f"{gs.COLOR_SHORT[targets[c]]} {shape}", px + 4, py + 16, (255, 255, 255),
                             0.45, 1, max_w=cs - 6)
                    count = len(maze.plates_at(c))
                    if count > 1:
                        gs.label(canvas, f"{count} plates", px + 4, py + 30, (255, 255, 255), 0.4, 1, max_w=cs - 6)
                    wall = maze.target_wall.get(c)
                    if wall is not None:  # แถบสีบอกผนังที่เป้าตั้งชิด
                        wall_dx, wall_dy = DIRS[wall]
                        mx, my = px + cs // 2 + wall_dx * (cs // 2 - 5), py + cs // 2 - wall_dy * (cs // 2 - 5)
                        half = max(4, cs // 4)
                        a, b = ((mx - half, my), (mx + half, my)) if wall_dx == 0 else ((mx, my - half), (mx, my + half))
                        cv2.line(canvas, a, b, gs.DISPLAY_COLORS[targets[c]], 5)
                    if c in self.picks:
                        gs.label(canvas, f"#{self.picks.index(c) + 1}", px + cs - 26, py + cs - 8, (80, 255, 255), 0.5, 2)
                    if c in results:
                        gs.label(canvas, results[c], px + 4, py + cs - 8, (255, 255, 255), 0.4, 1, max_w=cs - 30)
        for (a, b), status in list(maze.edges.items()):
            if status != -1 or not (inside(a) or inside(b)):
                continue
            (ax, ay), (bx, by) = a, b
            if ax == bx:  # กำแพงแนวนอน (ระหว่างช่องบน/ล่าง)
                if not wx <= ax < wx + ww:
                    continue
                top = max(ay, by)
                if not wy <= top <= wy + wh:
                    continue
                y_line = y0 + (wy + wh - top) * cs
                px = x0 + (ax - wx) * cs
                cv2.line(canvas, (px, y_line), (px + cs, y_line), WALL_COLOR, 4)
            else:        # กำแพงแนวตั้ง
                if not wy <= ay < wy + wh:
                    continue
                right = max(ax, bx)
                if not wx <= right <= wx + ww:
                    continue
                x_line = x0 + (right - wx) * cs
                row = wy + wh - 1 - ay
                cv2.line(canvas, (x_line, y0 + row * cs), (x_line, y0 + (row + 1) * cs), WALL_COLOR, 4)
        route = [c for c in m.route if inside(c)]
        if len(route) > 1:
            cv2.polylines(canvas, [np.array([center(c) for c in route], np.int32)], False, ROUTE_COLOR, 2)
        sx, sy = corner(m.start_cell)
        gs.label(canvas, "S", sx + cs - 16, sy + 18, GREEN, 0.55, 2)
        hx, hy = DIRS[m.start_heading]   # ลูกศรทิศที่ต้องวางหุ่นตอนเริ่ม
        scx, scy = sx + cs // 2, sy + cs // 2
        cv2.arrowedLine(canvas, (scx, scy), (scx + hx * cs // 3, scy - hy * cs // 3), GREEN, 1, tipLength=0.4)
        if self.goal is not None and maze.inside(self.goal) and inside(self.goal):
            gx, gy = corner(self.goal)
            cv2.rectangle(canvas, (gx + 3, gy + 3), (gx + cs - 3, gy + cs - 3), (255, 80, 255), 2)
            gs.label(canvas, "G", gx + cs - 16, gy + cs - 8, (255, 80, 255), 0.55, 2)
        # หุ่น: สามเหลี่ยมชี้ทิศตัวรถ + เส้นส้มคือทิศที่หัวกำลังมอง
        rx, ry = center(m.cell)
        hx, hy = DIRS[m.heading]
        r = max(8, cs // 4)
        tip = (rx + hx * r, ry - hy * r)
        left = (rx - hy * r * 0.7 - hx * r * 0.6, ry - hx * r * 0.7 + hy * r * 0.6)
        right = (rx + hy * r * 0.7 - hx * r * 0.6, ry + hx * r * 0.7 + hy * r * 0.6)
        cv2.fillPoly(canvas, [np.array([tip, left, right], np.int32)], ACCENT)
        cv2.polylines(canvas, [np.array([tip, left, right], np.int32)], True, TEXT, 1)
        if m.look_dir is not None:
            lx, ly = DIRS[m.look_dir]
            cv2.line(canvas, (rx, ry), (rx + lx * cs // 2, ry - ly * cs // 2), (0, 140, 255), 2)
        legend_x = x0 + ww * cs + 20
        gs.label(canvas, "LEGEND", legend_x, MAP_Y0 + 76, MUTED, 0.42, 2)
        for i, (text, color) in enumerate((("START", GREEN), ("GOAL", (255, 80, 255)),
                                           ("PICK TARGET", (80, 255, 255)), ("WALL", WALL_COLOR),
                                           ("BLOCKED", MUTED), ("ROUTE", ROUTE_COLOR))):
            cv2.circle(canvas, (legend_x + 5, MAP_Y0 + 99 + i * 31), 4, color, -1)
            gs.label(canvas, text, legend_x + 17, MAP_Y0 + 104 + i * 31, TEXT, 0.42, 1)
        gs.label(canvas, "Click map to set goal", legend_x, MAP_Y0 + 308, MUTED, 0.36, 1, max_w=CAM_W - legend_x - 12)

    def draw_panel(self, canvas):
        m = self.m
        px, bw = PANEL_X + 16, CANVAS_W - PANEL_X - 32
        half, third = (bw - 8) // 2, (bw - 16) // 3
        idle = not m.busy
        ready = idle and not m.pose_uncertain
        color = self.color_choices[self.color_filter]
        shape = self.shape_choices[self.shape_filter]
        filtered = color is not None or shape is not None
        exploring = m.busy and m.phase == "EXPLORE"
        confirm = time.monotonic() < self._clear_until

        for top, bottom in ((120, 244), (250, 359), (368, 534), (540, 584), (590, 660)):
            rounded_rect(canvas, px - 8, top, bw + 16, bottom - top, CARD)
        cv2.line(canvas, (px - 8, 120), (px + 32, 120), ACCENT, 3)
        cv2.line(canvas, (px - 8, 250), (px + 32, 250), BLUE, 3)
        cv2.line(canvas, (px - 8, 368), (px + 32, 368), GREEN, 3)

        gs.label(canvas, "ROBOMASTER  /  MAZE", px, 27, TEXT, 0.61, 2)
        rounded_rect(canvas, px + bw - 126, 8, 126, 25, (57, 48, 98) if m.armed else (50, 67, 47), 12)
        gs.label(canvas, "FIRE ARMED" if m.armed else "FIRE SAFE", px + bw - 113, 26,
                 RED if m.armed else GREEN, 0.46, 2)
        state = "STOPPED" if m.pose_uncertain else ("EXPLORE + FIRE" if m._explore_fire else "READY" if idle else m.phase)
        badge = RED if m.pose_uncertain or m.last_fault or m.armed else ACCENT if m.busy else GREEN
        rounded_rect(canvas, px, 40, bw, 30, (52, 49, 58))
        cv2.circle(canvas, (px + 16, 55), 5, badge, -1)
        gs.label(canvas, state, px + 30, 61, TEXT, 0.55, 2)
        if m.slam is not None:
            gs.label(canvas, f"SLAM {m.slam.confidence:.0%}", px + bw - 105, 61, ACCENT, 0.47, 2)
        gs.label(canvas, f"Cell {m.cell} / {DIR_NAMES[m.heading]}     Visited {len(m.maze.visited)}     Plates {m.maze.plate_count}",
                 px, 91, TEXT, 0.48, max_w=bw)
        hint = (m.last_fault or "Place robot at START, then click CONFIRM START (h)") if m.pose_uncertain else m.status
        gs.label(canvas, hint, px, 112, RED if m.pose_uncertain else MUTED, 0.46, max_w=bw)

        gs.label(canvas, "01   TARGET FILTER", px, 137, TEXT, 0.48, 2)
        # Direct choices keep the active filter visible without cycling through values.
        for choices, active, y, prefix in ((self.color_choices, self.color_filter, 145, "color"),
                                            (self.shape_choices, self.shape_filter, 183, "shape")):
            width = (bw - 8 * (len(choices) - 1)) // len(choices)
            for i, value in enumerate(choices):
                text = "ANY COLOR" if value is None and prefix == "color" else "ANY SHAPE" if value is None else "RECTANGLE" if value == "rect" else value.upper()
                tint = gs.dim(gs.DISPLAY_COLORS[value], 0.46) if prefix == "color" and value else BUTTON
                self.button(canvas, px + i * (width + 8), y, width, 32, text,
                               f"{prefix}:{value or 'any'}", tint, enabled=idle, selected=i == active)
        target = f"{(color or 'ANY COLOR').upper()} + {(shape or 'ANY SHAPE').upper()}"
        gs.label(canvas, ("FILTER  " + target) if filtered else "Select a color or shape to enable fire", px, 236, GREEN if filtered else ACCENT, 0.48, 2)

        gs.label(canvas, "02   EXPLORE", px, 258, TEXT, 0.48, 2)
        self.button(canvas, px, 266, half, 46, "EXPLORE + FIRE  [E]", "explore", RED, enabled=ready and filtered)
        self.button(canvas, px + half + 8, 266, half, 46, "MAP ONLY  [M]", "map_only", BLUE, enabled=ready)
        self.button(canvas, px, 320, bw, 32, "FINISH EXPLORATION AT NEXT CELL (f)", "finish",
                       enabled=exploring or (ready and not m.maze.explored))

        gs.label(canvas, "03   USE MAP" if m.maze.explored else "03   USE MAP  /  COMPLETE EXPLORATION FIRST", px, 378, TEXT, 0.48, 2)
        self.button(canvas, px, 386, half, 42, "RUN + FIRE  [ENTER]", "shoot", RED, enabled=ready and m.maze.explored)
        self.button(canvas, px + half + 8, 386, half, 42, "DRY RUN  [D]", "dry", BLUE, enabled=ready and m.maze.explored)
        self.button(canvas, px, 436, bw, 28, "HIDE ROUTE OPTIONS" if self.show_options else "SHOW ROUTE OPTIONS", "options")
        if self.show_options:
            self.button(canvas, px, 472, third, 30, f"START: {START_MODES[self.start_mode]} (1)", "opt_start", enabled=idle)
            self.button(canvas, px + third + 8, 472, third, 30, f"END: {END_MODES[self.end_mode]}" + (f" {self.goal}" if END_MODES[self.end_mode] == "GO TO GOAL" and self.goal else "") + " (2)", "opt_end", enabled=idle)
            self.button(canvas, px + 2 * (third + 8), 472, third, 30, f"ORDER: {ORDER_MODES[self.order_mode]} (3)", "opt_order", enabled=idle)
            pick = " > ".join(f"{i + 1}.{m.maze.target_label(c)}" for i, c in enumerate(self.picks) if c in m.maze.targets)
            gs.label(canvas, "Picks: " + (pick or "click targets on map"), px, 524, MUTED, 0.43, max_w=bw - 180)
            self.button(canvas, px + bw - 168, 510, 80, 26, "UNDO (u)", "undo", enabled=idle)
            self.button(canvas, px + bw - 80, 510, 80, 26, "CLEAR", "clear_picks", enabled=idle)
        else:
            gs.label(canvas, "ROUTE   " + START_MODES[self.start_mode] + "  /  " + END_MODES[self.end_mode], px, 489, MUTED, 0.44, max_w=bw)
            gs.label(canvas, "Click a target to pick order; click an empty cell to set goal", px, 516, MUTED, 0.44, max_w=bw)

        self.button(canvas, px, 544, third, 32, "CONFIRM START (h)", "home",
                       ACCENT if m.pose_uncertain else BUTTON, enabled=idle)
        self.button(canvas, px + third + 8, 544, third, 32, "CENTER CAMERA (c)", "center", enabled=idle)
        self.button(canvas, px + 2 * (third + 8), 544, third, 32,
                       "CONFIRM ERASE (y)" if confirm else "ERASE MAP (n)", "confirm_clear" if confirm else "clear",
                       RED if confirm else BUTTON, enabled=idle)
        gs.label(canvas, "LATEST EVENTS", px, 607, MUTED, 0.42, 2)
        for i, line in enumerate(list(m.logs)[-3:]):
            gs.label(canvas, line, px, 627 + i * 15, TEXT, 0.43, max_w=bw)
        self.button(canvas, px, 668, bw - 128, 40, "STOP NOW  [X / SPACE]", "stop", RED)
        self.button(canvas, px + bw - 120, 668, 120, 40, "QUIT  [Q]", "quit", BUTTON)
