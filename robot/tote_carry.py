#!/usr/bin/env python3
"""
G1 tote carry -- hold the right arm in a bag-carrying pose while walking.

The robot stays in its normal locomotion ("running") mode and is driven with
the wireless remote exactly as usual. This script only overrides the ARMS via
the `rt/arm_sdk` topic: the stock locomotion controller blends our arm targets
in with weight `motor_cmd[29].q` (0 = loco owns the arms, 1 = we do) and keeps
balancing the legs + waist itself. Same mechanism as Unitree's xr_teleoperate
"motion mode".

Pose: right forearm bent ~100 deg and pointing forward, tilted slightly up so
the tote handles slide into the crook of the elbow, upper arm a little out from
the body so the bag hangs clear of the thigh. The left arm is held where it was
when the script started.

Sequence:  wait for rt/lowstate -> capture current arm pose -> fade arm_sdk
weight 0->1 holding that pose -> move the right arm (speed-limited) into the
tote pose -> HOLD until Ctrl-C / SIGTERM -> move the right arm back -> fade the
weight 1->0 (arms handed back to the locomotion controller).

Run on the Orin, in the greeter image (has unitree_sdk2py + cyclonedds):
    docker run --rm -it --network host \
        -v ~/g1demo/tote_carry.py:/app/tote_carry.py:ro \
        --entrypoint python3 g1-greeter-shadow:latest \
        /app/tote_carry.py eth0 --check          # read-only: print pose + plan
    ... same without --check to actually move the arm.

ALWAYS do a --check first with the robot standing in locomotion mode: it prints
the measured arm joints and the planned target. A standing G1 reads elbow
~ +0.8..1.0 rad; if yours reads negative the sign convention differs from this
script's assumption -- do not run the motion until the pose is corrected.

Options (all optional):
    --check              read-only: print state + plan, send nothing
    --pose a,b,c,d,e,f,g override right-arm target (rad): shoulder pitch, roll,
                         yaw, elbow, wrist roll, pitch, yaw
    --kp N / --kd N      shoulder+elbow gains (default 60 / 1.5)
    --speed R            max joint speed for the move, rad/s (default 0.6)
    --hold-secs S        release automatically after S seconds (default: forever)
"""

import argparse
import math
import signal
import sys
import threading
import time

# ----- G1 29-dof joint indices (rt/lowstate / rt/arm_sdk motor order) -------
LEFT_ARM  = [15, 16, 17, 18, 19, 20, 21]   # sh pitch, sh roll, sh yaw, elbow, wr roll, wr pitch, wr yaw
RIGHT_ARM = [22, 23, 24, 25, 26, 27, 28]
ARM_JOINTS = LEFT_ARM + RIGHT_ARM
# arm_sdk at weight 1 also takes the WAIST: any of these joints we leave at
# kp=0 goes limp (the robot folded at the waist and fell on 2026-10-02). So the
# waist is always commanded too, held stiff at where it was when we started.
WAIST = [12, 13, 14]                       # yaw, roll, pitch
KP_WAIST, KD_WAIST = 200.0, 6.0      # what the loco controller itself uses
WEIGHT_IDX = 29                            # kNotUsedJoint: arm_sdk blend weight lives in its .q

JOINT_NAMES = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
               "wrist_roll", "wrist_pitch", "wrist_yaw"]

# Right-arm URDF limits (rad), same order as JOINT_NAMES. A small margin is
# applied on top so a typo in --pose can never drive a joint into its stop.
RIGHT_LIMITS = [(-3.0892, 2.6704), (-2.2515, 1.5882), (-2.618, 2.618),
                (-1.0472, 2.0944), (-1.9722, 1.9722), (-1.6144, 1.6144),
                (-1.6144, 1.6144)]
LIMIT_MARGIN = 0.1

# Conventions (match the stock standing pose: shoulder pitch +0.35, right roll
# -0.16, elbow +0.87): +shoulder pitch swings the arm back, -right shoulder roll
# swings it out, +elbow bends it.
TOTE_POSE = [-0.10,   # shoulder pitch: upper arm a touch forward
             -0.25,   # shoulder roll: out from the body, bag clears the thigh
              0.00,   # shoulder yaw
              1.75,   # elbow ~100 deg: forearm forward, tilted up a little
              0.00, 0.00, 0.00]   # wrist neutral

KP_ARM, KD_ARM     = 60.0, 1.5
KP_WRIST, KD_WRIST = 40.0, 1.5
CONTROL_DT   = 0.02    # 50 Hz -- arm_sdk needs a steady stream
FADE_SECS    = 1.0     # weight ramp in/out
MAX_SPEED    = 0.6     # rad/s, cap on the arm move
MIN_MOVE_SECS = 1.5
LOWSTATE_TIMEOUT = 3.0
TEACH_KD = 1.0         # right-arm damping while it is limp in --teach


# ============================ pure helpers (unit-tested) =====================
def clamp_pose(pose):
    """Clamp a right-arm pose into the joint limits (minus a safety margin)."""
    if len(pose) != 7:
        raise ValueError(f"pose needs 7 joint values, got {len(pose)}")
    return [min(max(q, lo + LIMIT_MARGIN), hi - LIMIT_MARGIN)
            for q, (lo, hi) in zip(pose, RIGHT_LIMITS)]


def move_duration(start, target, max_speed=MAX_SPEED, min_secs=MIN_MOVE_SECS):
    """Seconds needed so no joint exceeds max_speed on a smoothstep move.
    Smoothstep peaks at 1.5x the average speed, hence the factor."""
    if max_speed <= 0:
        raise ValueError("max_speed must be > 0")
    biggest = max((abs(t - s) for s, t in zip(start, target)), default=0.0)
    return max(min_secs, 1.5 * biggest / max_speed)


def smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def lerp_pose(start, target, s):
    a = smoothstep(s)
    return [q0 + a * (q1 - q0) for q0, q1 in zip(start, target)]


def parse_pose(text):
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 7:
        raise argparse.ArgumentTypeError("--pose needs 7 comma-separated values")
    return vals


def format_pose(pose):
    return "  ".join(f"{n}={q:+.2f}" for n, q in zip(JOINT_NAMES, pose))


# ============================ robot side =====================================
class ToteCarry:
    def __init__(self, iface, kp, kd):
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                  ChannelPublisher, ChannelSubscriber)
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, iface)
        self._lock = threading.Lock()
        self._state = None
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        # The loco controller's own motor commands (read-only, diagnostics).
        self._loco = None
        self._loco_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self._loco_sub.Init(self._on_loco, 10)
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self.kp, self.kd = kp, kd
        self.weight = 0.0           # last blend weight actually sent
        self.waist_q = None         # waist hold targets, captured before taking control

    def _on_state(self, msg):
        with self._lock:
            self._state = msg

    def _on_loco(self, msg):
        with self._lock:
            self._loco = msg

    def loco_summary(self):
        """rt/lowcmd waist yaw + right elbow (q, kp) -- shows whether that topic
        is the loco output before or after our arm_sdk blend."""
        with self._lock:
            m = self._loco
        if m is None:
            return "rt/lowcmd: none"
        w, e = m.motor_cmd[12], m.motor_cmd[25]
        return (f"rt/lowcmd waist_yaw q={w.q:+.3f} kp={w.kp:.0f} | "
                f"r_elbow q={e.q:+.3f} kp={e.kp:.0f}")

    def wait_state(self, timeout=LOWSTATE_TIMEOUT):
        t_end = time.time() + timeout
        while time.time() < t_end:
            with self._lock:
                if self._state is not None:
                    return self._state
            time.sleep(0.05)
        return None

    def arm_q(self):
        """Measured positions of the 14 arm joints (left then right)."""
        with self._lock:
            st = self._state
        return [st.motor_state[j].q for j in ARM_JOINTS]

    def waist(self):
        with self._lock:
            st = self._state
        return [st.motor_state[j].q for j in WAIST]

    def mode(self):
        with self._lock:
            return getattr(self._state, "mode_machine", None)

    def send(self, arm_targets, weight, right_gain=1.0, right_kd=None):
        """One arm_sdk frame: 14 arm targets + held waist + blend weight.
        right_gain scales the right arm's kp (0 = limp, only damping);
        right_kd overrides its kd."""
        if self.waist_q is None:
            raise RuntimeError("waist hold not captured -- refusing to send arm_sdk")
        for j, q in zip(WAIST, self.waist_q):
            m = self._cmd.motor_cmd[j]
            m.q, m.dq, m.tau, m.kp, m.kd = q, 0.0, 0.0, KP_WAIST, KD_WAIST
        for j, q in zip(ARM_JOINTS, arm_targets):
            wrist = (j - 15) % 7 >= 4
            m = self._cmd.motor_cmd[j]
            m.q, m.dq, m.tau = q, 0.0, 0.0
            m.kp = KP_WRIST if wrist else self.kp
            m.kd = KD_WRIST if wrist else self.kd
            if j in RIGHT_ARM:
                m.kp *= right_gain
                if right_kd is not None:
                    m.kd = right_kd
        self.weight = min(max(weight, 0.0), 1.0)
        self._cmd.motor_cmd[WEIGHT_IDX].q = self.weight
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)


def _ticks(secs):
    return max(1, int(round(secs / CONTROL_DT)))


def _run_phase(robot, secs, frame_fn, stop=None):
    """Stream frames at 50 Hz for `secs`; frame_fn(s in 0..1) -> (targets, weight).
    Returns early (False) if `stop` gets set."""
    n = _ticks(secs)
    t_next = time.perf_counter()
    for i in range(1, n + 1):
        if stop is not None and stop.is_set():
            return False
        robot.send(*frame_fn(i / n))
        t_next += CONTROL_DT
        time.sleep(max(0.0, t_next - time.perf_counter()))
    return True


def still_pose(samples, window, tol=0.02):
    """Mean of the latest `window` consecutive samples in which no joint moved
    more than `tol` rad, or None if the arm never sat still that long."""
    for end in range(len(samples), window - 1, -1):
        chunk = samples[end - window:end]
        cols = list(zip(*chunk))
        if all(max(c) - min(c) <= tol for c in cols):
            return [sum(c) / len(c) for c in cols]
    return None


def record(robot, secs, rate=50):
    """Read-only pose capture: nothing is published."""
    print(f"[record] logging arms for {secs:.0f}s (mode {robot.mode()}) -- move the "
          "right arm into place and hold it still for ~2s")
    samples, t_end, n = [], time.time() + secs, 0
    while time.time() < t_end:
        q = robot.arm_q()
        samples.append(q[7:])
        if n % rate == 0:
            print(f"[record] right: {format_pose(q[7:])}", flush=True)
        n += 1
        time.sleep(1.0 / rate)
    pose = still_pose(samples, 2 * rate)
    if pose is None:
        print("[record] arm never held still for 2s -- last sample:")
        pose = samples[-1]
    print(f"[record] RIGHT POSE: {format_pose(pose)}")
    print("[record] --pose " + ",".join(f"{q:.3f}" for q in pose))
    print(f"[record] left (for reference): {format_pose(robot.arm_q()[:7])}")
    return 0


def teach(robot, start, secs, speed, rate=50):
    """Limp right arm (kp 0, light damping) for `secs` while the left arm and
    waist stay stiff and the legs stay with the loco controller. Records the
    last still pose, re-stiffens the arm where it was left, then puts it back."""
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    start_right = start[7:]
    hold = list(start)
    samples = []
    print("[teach] taking the arms (weight 0 -> 1, holding current pose)")
    _run_phase(robot, FADE_SECS, lambda s: (hold, smoothstep(s)), stop)
    try:
        print("[teach] HOLD THE RIGHT ARM -- it goes soft in 1s")
        _run_phase(robot, 1.0, lambda s: (hold, 1.0, 1.0 - smoothstep(s),
                                          KD_ARM + (TEACH_KD - KD_ARM) * s), stop)
        print(f"[teach] right arm is limp for {secs:.0f}s -- move it into the carry "
              "pose and hold it still for ~2s", flush=True)
        t_end, n = time.time() + secs, 0
        while not stop.is_set() and time.time() < t_end:
            robot.send(hold, 1.0, 0.0, TEACH_KD)
            q = robot.arm_q()[7:]
            samples.append(q)
            if n % rate == 0:
                print(f"[teach] right: {format_pose(q)}", flush=True)
                print(f"[teach]   {robot.loco_summary()}", flush=True)
            n += 1
            time.sleep(1.0 / rate)
    finally:
        stop_now = threading.Event()
        here = robot.arm_q()[7:]
        hold[7:] = here
        print("[teach] re-stiffening the arm where it is (1s)")
        _run_phase(robot, 1.0, lambda s: (hold, robot.weight, smoothstep(s),
                                          TEACH_KD + (KD_ARM - TEACH_KD) * s), stop_now)
        pose = still_pose(samples, 2 * rate) if samples else None
        if pose is None:
            print("[teach] arm never held still for 2s -- no pose recorded")
        else:
            print(f"[teach] RIGHT POSE: {format_pose(pose)}")
            print("[teach] --pose " + ",".join(f"{q:.3f}" for q in pose))
        back = move_duration(here, start_right, speed)
        print(f"[teach] returning right arm ({back:.1f}s), then releasing")

        def to_start(s):
            hold[7:] = lerp_pose(here, start_right, s)
            return hold, robot.weight
        _run_phase(robot, back, to_start, stop_now)
        w0 = robot.weight
        _run_phase(robot, FADE_SECS, lambda s: (hold, w0 * (1.0 - smoothstep(s))), stop_now)
        robot.send(hold, 0.0)
        print("[teach] released.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hold the G1 right arm in a tote-carry pose")
    ap.add_argument("iface", nargs="?", default="eth0")
    ap.add_argument("--check", action="store_true", help="read-only: print state + plan")
    ap.add_argument("--pose", type=parse_pose, default=TOTE_POSE)
    ap.add_argument("--kp", type=float, default=KP_ARM)
    ap.add_argument("--kd", type=float, default=KD_ARM)
    ap.add_argument("--speed", type=float, default=MAX_SPEED)
    ap.add_argument("--hold-secs", type=float, default=0.0)
    ap.add_argument("--teach", type=float, default=0.0, metavar="SECS",
                    help="robot keeps standing; right arm goes limp (damped) for "
                         "SECS so you can pose it by hand, then the still pose is "
                         "printed and the arm is put back")
    ap.add_argument("--record", type=float, default=0.0, metavar="SECS",
                    help="read-only: log both arms for SECS (pose the arm by hand in "
                         "damped mode) and print the last still pose as a --pose value")
    args = ap.parse_args(argv)

    target_right = clamp_pose(args.pose)
    if target_right != list(args.pose):
        print(f"[tote] pose clamped to joint limits: {format_pose(target_right)}")

    robot = ToteCarry(args.iface, args.kp, args.kd)
    if robot.wait_state() is None:
        print(f"[tote] no rt/lowstate on {args.iface} within {LOWSTATE_TIMEOUT}s -- "
              "is the robot on and the interface right? Nothing sent.")
        return 2

    if args.record > 0:
        return record(robot, args.record)

    start = robot.arm_q()
    robot.waist_q = robot.waist()
    print(f"[tote] waist hold: {['%+.2f' % q for q in robot.waist_q]}")
    start_left, start_right = start[:7], start[7:]
    move_secs = move_duration(start_right, target_right, args.speed)
    print(f"[tote] mode_machine={robot.mode()}")
    print(f"[tote] left  now : {format_pose(start_left)}")
    print(f"[tote] right now : {format_pose(start_right)}")
    print(f"[tote] right goal: {format_pose(target_right)}  (move {move_secs:.1f}s)")
    if start_right[3] < 0.2:
        print("[tote] WARNING: right elbow reads < 0.2 rad. A standing G1 normally "
              "reads ~0.8-1.0 -- check the sign convention before moving.")
    if args.check:
        print("[tote] --check: nothing sent.")
        return 0
    if args.teach > 0:
        return teach(robot, start, args.teach, args.speed)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    hold = list(start)   # what we are currently commanding (left + right)
    print("[tote] taking the arms (weight 0 -> 1, holding current pose)")
    took = _run_phase(robot, FADE_SECS, lambda s: (hold, smoothstep(s)), stop)
    try:
        if took:
            print("[tote] moving right arm into tote pose")

            def to_tote(s):
                hold[7:] = lerp_pose(start_right, target_right, s)
                return hold, 1.0
            if _run_phase(robot, move_secs, to_tote, stop):
                print("[tote] holding -- drive with the remote as usual. "
                      "Ctrl-C to lower the arm and release.")
                t_end = time.time() + args.hold_secs if args.hold_secs > 0 else math.inf
                while not stop.is_set() and time.time() < t_end:
                    _run_phase(robot, 2.0, lambda s: (hold, 1.0), stop)
                    meas = robot.arm_q()[7:]
                    err = max(abs(m - t) for m, t in zip(meas, target_right))
                    print(f"[tote] right meas: {format_pose(meas)}  "
                          f"(max err {err:.2f} rad, mode {robot.mode()})")
    finally:
        # Always hand the arms back, even after Ctrl-C mid-move or an error.
        stop_now = threading.Event()   # the shutdown path itself is not interruptible
        here = list(hold[7:])
        back_secs = move_duration(here, start_right, args.speed)
        print(f"[tote] lowering right arm ({back_secs:.1f}s)")

        def to_start(s):
            hold[7:] = lerp_pose(here, start_right, s)
            return hold, robot.weight
        _run_phase(robot, back_secs, to_start, stop_now)
        print("[tote] releasing arms to the locomotion controller (weight 1 -> 0)")
        w0 = robot.weight          # may be < 1 if stopped mid fade-in
        _run_phase(robot, FADE_SECS, lambda s: (hold, w0 * (1.0 - smoothstep(s))), stop_now)
        robot.send(hold, 0.0)
        print("[tote] released.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
