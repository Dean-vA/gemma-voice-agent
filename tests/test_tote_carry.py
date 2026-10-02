"""Pure pose/timing helpers of robot/tote_carry.py (no SDK, no robot).

Run:  pytest tests/test_tote_carry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robot"))
import tote_carry as tc  # noqa: E402

# Right arm as measured on the robot standing in locomotion mode (2026-10-02).
STANDING_RIGHT = [0.25, -0.30, 0.08, 0.78, 0.0, 0.01, 0.0]


def test_default_pose_is_within_limits():
    assert tc.clamp_pose(tc.TOTE_POSE) == tc.TOTE_POSE


def test_clamp_pose_keeps_margin_from_stops():
    clamped = tc.clamp_pose([10.0, -10.0, 0, 10.0, 0, 0, 0])
    assert clamped[0] == pytest.approx(tc.RIGHT_LIMITS[0][1] - tc.LIMIT_MARGIN)
    assert clamped[1] == pytest.approx(tc.RIGHT_LIMITS[1][0] + tc.LIMIT_MARGIN)
    assert clamped[3] == pytest.approx(tc.RIGHT_LIMITS[3][1] - tc.LIMIT_MARGIN)


def test_clamp_pose_rejects_wrong_length():
    with pytest.raises(ValueError):
        tc.clamp_pose([0.0] * 6)


def test_move_duration_respects_speed_cap():
    secs = tc.move_duration(STANDING_RIGHT, tc.TOTE_POSE, max_speed=0.6)
    # Smoothstep peak speed is 1.5 * delta / T; sample it to be sure.
    n = 1000
    dt = secs / n
    peak = max(
        abs(a - b) / dt
        for i in range(n)
        for a, b in zip(tc.lerp_pose(STANDING_RIGHT, tc.TOTE_POSE, (i + 1) / n),
                        tc.lerp_pose(STANDING_RIGHT, tc.TOTE_POSE, i / n))
    )
    assert peak <= 0.6 + 1e-3


def test_move_duration_has_floor():
    assert tc.move_duration(STANDING_RIGHT, STANDING_RIGHT) == tc.MIN_MOVE_SECS


def test_lerp_pose_endpoints():
    assert tc.lerp_pose(STANDING_RIGHT, tc.TOTE_POSE, 0.0) == STANDING_RIGHT
    assert tc.lerp_pose(STANDING_RIGHT, tc.TOTE_POSE, 1.0) == pytest.approx(tc.TOTE_POSE)


def test_parse_pose():
    assert tc.parse_pose("0,-0.2,0,1.6,0,0,0") == [0, -0.2, 0, 1.6, 0, 0, 0]
    with pytest.raises(Exception):
        tc.parse_pose("1,2,3")


def test_still_pose_takes_latest_still_window():
    moving = [[0.1 * i] * 7 for i in range(20)]
    still = [[1.0] * 7 for _ in range(10)]
    nudge = [[1.5] * 7]
    assert tc.still_pose(moving + still + nudge, 10) == pytest.approx([1.0] * 7)
    assert tc.still_pose(moving, 10) is None


def test_waist_is_held_whenever_arm_sdk_is_sent():
    # Leaving the waist out of an arm_sdk frame makes it limp (robot fell).
    assert tc.WAIST == [12, 13, 14] and tc.KP_WAIST > 0
