"""Bounded sideways correction using stationary, level ToF wall readings."""

from typing import Optional

MAX_ADJACENT_MM = 450
MAX_SHIFT_MM = 60
MIN_CLEARANCE_MM = 140
MIN_ERROR_MM = 20
MAX_WIDTH_ERROR_MM = 120


def center_shift_mm(left_mm: Optional[float], right_mm: Optional[float],
                    wall_offset_mm: Optional[float]) -> float:
    """Return desired chassis motion toward the left in mm; zero means skip."""
    left = left_mm if left_mm is not None and 20 <= left_mm <= MAX_ADJACENT_MM else None
    right = right_mm if right_mm is not None and 20 <= right_mm <= MAX_ADJACENT_MM else None
    if left is not None and right is not None:
        if wall_offset_mm is not None and abs(left + right - 2 * wall_offset_mm) > MAX_WIDTH_ERROR_MM:
            return 0.0
        shift = (left - right) / 2
    elif wall_offset_mm is not None and left is not None:
        shift = left - wall_offset_mm
    elif wall_offset_mm is not None and right is not None:
        shift = wall_offset_mm - right
    else:
        return 0.0
    if abs(shift) < MIN_ERROR_MM:
        return 0.0
    shift = max(-MAX_SHIFT_MM, min(MAX_SHIFT_MM, shift))
    if shift > 0 and left is not None and left - shift < MIN_CLEARANCE_MM:
        return 0.0
    if shift < 0 and right is not None and right + shift < MIN_CLEARANCE_MM:
        return 0.0
    return shift
