"""Pure-Python metrics shared by simulator and analysis."""
import math

def wrapped_yaw_error_deg(actual_deg: float, target_deg: float) -> float:
    return abs((actual_deg - target_deg + 180.0) % 360.0 - 180.0)

def position_error_m(actual_xy, target_xy) -> float:
    return math.hypot(actual_xy[0] - target_xy[0], actual_xy[1] - target_xy[1])

def is_success(position_error: float, yaw_error: float, position_tol=0.02, yaw_tol=5.0) -> bool:
    return position_error <= position_tol and yaw_error <= yaw_tol
