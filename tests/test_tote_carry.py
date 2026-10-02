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


# ---------------------------------------------------------------- --daemon
import threading  # noqa: E402
import time  # noqa: E402


def test_combo_edge_needs_both_keys_newly_held():
    F1, F2 = tc.KEY_F1, tc.KEY_F2
    assert tc.combo_edge(0, F1 | F2)
    assert tc.combo_edge(F1, F1 | F2)          # F1 first, then F2 joins
    assert not tc.combo_edge(F1 | F2, F1 | F2)  # still held
    assert not tc.combo_edge(0, F1)
    assert not tc.combo_edge(0, F2)


def test_release_reason():
    assert tc.release_reason(501, 0.01) is None
    assert tc.release_reason(None, 0.01) is None       # API read timeout: keep holding
    assert tc.release_reason(801, 0.01) == "mode"
    assert tc.release_reason(812, 0.01) == "mode"
    assert tc.release_reason(1, 0.01) == "fast"        # damp
    assert tc.release_reason(0, 0.01) == "fast"        # zero torque
    assert tc.release_reason(501, 5.0) == "fast"       # lowstate stopped


class FakeRobot:
    def __init__(self):
        self.q = list(STANDING_RIGHT) + list(STANDING_RIGHT)
        self.fsm = 501
        self._keys = 0
        self.weight = 0.0
        self.waist_q = None
        self.frames = []

    def listen_remote(self): pass
    def keys(self): return self._keys
    def fsm_id(self): return self.fsm
    def state_age(self): return 0.01
    def arm_q(self): return list(self.q)
    def waist(self): return [0.0, 0.0, 0.0]

    def send(self, targets, weight, gain=1.0, kd=None):
        assert self.waist_q is not None and len(targets) == 14
        self.weight = min(max(weight, 0.0), 1.0)
        self.frames.append((list(targets), self.weight))

    def tap(self, keys, secs=0.15):
        self._keys = keys
        time.sleep(secs)
        self._keys = 0


def _fast_timing(monkeypatch):
    monkeypatch.setattr(tc, "CONTROL_DT", 0.002)
    monkeypatch.setattr(tc, "FADE_SECS", 0.05)
    monkeypatch.setattr(tc, "MIN_MOVE_SECS", 0.05)
    monkeypatch.setattr(tc, "FSM_POLL_SECS", 0.01)
    monkeypatch.setattr(tc, "STEP_SPEED", 50.0)


def _run_daemon(monkeypatch, robot, script):
    _fast_timing(monkeypatch)
    stop = threading.Event()
    monkeypatch.setattr(tc, "_stop_on_signals", lambda: stop)

    def drive():
        try:
            script(robot)
        finally:
            stop.set()
    threading.Thread(target=drive, daemon=True).start()
    tc.daemon(robot, tc.TOTE_POSES, speed=50.0)  # fast moves for the test


def test_daemon_cycles_pose1_pose2_then_lowers(monkeypatch):
    seen = {}

    def script(r):
        time.sleep(0.1)
        r.tap(tc.KEY_COMBO)                     # off -> pose 1
        time.sleep(0.4)
        seen["p1"] = (r.weight, r.frames[-1][0][7:])
        r.tap(tc.KEY_COMBO)                     # pose 1 -> pose 2
        time.sleep(0.4)
        seen["p2"] = (r.weight, r.frames[-1][0][7:])
        r.tap(tc.KEY_COMBO)                     # pose 2 -> lowered, off
        time.sleep(0.6)
    r = FakeRobot()
    _run_daemon(monkeypatch, r, script)
    assert seen["p1"][0] == 1.0 and seen["p1"][1] == pytest.approx(tc.TOTE_POSE)
    assert seen["p2"][0] == 1.0 and seen["p2"][1] == pytest.approx(tc.TOTE_POSE_2)
    assert r.weight == 0.0
    assert r.frames[-1][0][7:] == pytest.approx(STANDING_RIGHT)   # lowered first


def test_daemon_ignores_press_during_move(monkeypatch):
    seen = {}

    def script(r):
        time.sleep(0.1)
        r.tap(tc.KEY_COMBO, secs=0.02)          # off -> pose 1 (move is slow here)
        time.sleep(0.03)
        r.tap(tc.KEY_COMBO, secs=0.02)          # mid-move: must not skip to pose 2
        time.sleep(1.0)
        seen["after"] = (r.weight, r.frames[-1][0][7:])
    r = FakeRobot()
    _fast_timing(monkeypatch)
    monkeypatch.setattr(tc, "MIN_MOVE_SECS", 0.5)
    stop = threading.Event()
    monkeypatch.setattr(tc, "_stop_on_signals", lambda: stop)

    def drive():
        try:
            script(r)
        finally:
            stop.set()
    threading.Thread(target=drive, daemon=True).start()
    tc.daemon(r, tc.TOTE_POSES, speed=50.0)
    assert seen["after"][0] == 1.0
    assert seen["after"][1] == pytest.approx(tc.TOTE_POSE)    # still pose 1


def test_pose2_is_within_limits():
    assert tc.clamp_pose(tc.TOTE_POSE_2) == tc.TOTE_POSE_2


def test_daemon_ignores_combo_outside_walk_mode(monkeypatch):
    def script(r):
        time.sleep(0.1)
        r.tap(tc.KEY_COMBO)
        time.sleep(0.2)
    r = FakeRobot()
    r.fsm = 801
    _run_daemon(monkeypatch, r, script)
    assert r.frames == []                       # never published


def test_daemon_ignores_f1_or_f2_alone(monkeypatch):
    def script(r):
        time.sleep(0.1)
        r.tap(tc.KEY_F1)
        r.tap(tc.KEY_F2)
        time.sleep(0.1)
    r = FakeRobot()
    _run_daemon(monkeypatch, r, script)
    assert r.frames == []


def test_daemon_lowers_when_robot_leaves_walk_mode(monkeypatch):
    def script(r):
        time.sleep(0.1)
        r.tap(tc.KEY_COMBO)
        time.sleep(0.4)
        r.fsm = 801
        time.sleep(0.6)
    r = FakeRobot()
    _run_daemon(monkeypatch, r, script)
    assert r.weight == 0.0
    assert r.frames[-1][0][7:] == pytest.approx(STANDING_RIGHT)


def test_daemon_drops_fast_when_robot_goes_soft(monkeypatch):
    def script(r):
        time.sleep(0.1)
        r.tap(tc.KEY_COMBO)
        time.sleep(0.4)
        r.q[7:] = [0.0] * 7                     # arm sagging as the body goes soft
        r.fsm = 1                               # damp
        time.sleep(0.4)
    r = FakeRobot()
    _run_daemon(monkeypatch, r, script)
    assert r.weight == 0.0
    assert r.frames[-1][0][7:] == pytest.approx([0.0] * 7)   # let go where it was


def test_daemon_releases_on_stop(monkeypatch):
    def script(r):
        time.sleep(0.1)
        r.tap(tc.KEY_COMBO)
        time.sleep(0.4)                         # still held when stop is set
    r = FakeRobot()
    _run_daemon(monkeypatch, r, script)
    assert r.weight == 0.0


def test_still_segments_lists_each_held_pose_once():
    a, b = [0.0] * 7, [1.0] * 7
    move = [[0.1 * i] * 7 for i in range(1, 10)]
    samples = [a] * 10 + move + [b] * 10 + [[1.01] * 7] * 5
    segs = tc.still_segments(samples, 5)
    assert [i for i, _ in segs] == [0, 19]
    assert segs[0][1] == pytest.approx(a) and segs[1][1] == pytest.approx(b, abs=0.01)
