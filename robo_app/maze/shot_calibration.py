"""Range-dependent correction for camera/blaster alignment in the maze."""

import json
import math
from pathlib import Path
from typing import List, Tuple


CALIBRATION_FILE = Path(__file__).resolve().parents[2] / "shot_calibration.json"


def load_shot_calibration(path: Path = CALIBRATION_FILE) -> List[Tuple[float, float, float]]:
    """Return (target distance m, target-image yaw deg, pitch deg) observations."""
    if not path.exists():
        return []
    try:
        samples = json.loads(path.read_text(encoding="utf-8"))["samples"]
        if not isinstance(samples, list):
            raise ValueError("samples must be a list")
        points = []
        for sample in samples:
            distance = float(sample["distance_m"])
            right = float(sample["miss_right_cm"])
            up = float(sample["miss_up_cm"])
            if not all(math.isfinite(v) for v in (distance, right, up)):
                raise ValueError("non-finite sample")
            if not (0.15 <= distance <= 1.5 and abs(right) <= 10 and abs(up) <= 10):
                raise ValueError("sample outside supported target/impact range")
            points.append((distance, math.degrees(math.atan2(right / 100, distance)),
                           math.degrees(math.atan2(up / 100, distance))))
        if len(points) > 10 or len({p[0] for p in points}) != len(points):
            raise ValueError("use at most 10 distinct distances")
        return sorted(points)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[CONFIG] Ignore invalid shot calibration: {exc}")
        return []


def shot_offset(points: List[Tuple[float, float, float]], distance: float) -> Tuple[float, float]:
    """Interpolate measured camera-to-impact correction, clamping outside samples."""
    if not points:
        return 0.0, 0.0
    if distance <= points[0][0]:
        return points[0][1:]
    for lo, hi in zip(points, points[1:]):
        if distance <= hi[0]:
            weight = (distance - lo[0]) / (hi[0] - lo[0])
            return (lo[1] + weight * (hi[1] - lo[1]),
                    lo[2] + weight * (hi[2] - lo[2]))
    return points[-1][1:]
