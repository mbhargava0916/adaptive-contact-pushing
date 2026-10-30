import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adaptive_contact_pushing.metrics import wrapped_yaw_error_deg, position_error_m, is_success

def test_yaw_wrap():
    assert wrapped_yaw_error_deg(179, -179) == 2

def test_position():
    assert position_error_m((0, 0), (0.03, 0.04)) == 0.05

def test_success():
    assert is_success(0.01, 3)
    assert not is_success(0.03, 3)
