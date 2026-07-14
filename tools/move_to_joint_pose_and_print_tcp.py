from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from deploy.config import load_config
from deploy.devices.piper_robot import PiperRobot
from deploy.targets import TRAINING_START_DEG


CONFIRM_TEXT = "MOVE_PIPER_TO_JOINTS_AND_PRINT_TCP"
DEFAULT_TARGET_DEG = TRAINING_START_DEG.copy()


def quat_wxyz_from_euler_xyz(euler_xyz_rad: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_euler("xyz", euler_xyz_rad).as_quat()
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def print_tcp_snapshot(robot: PiperRobot, prefix: str = "tcp") -> None:
    snapshot = robot.read_feedback_snapshot()
    position = np.asarray(snapshot["model_tcp"]["position_m"], dtype=np.float64)
    euler = np.asarray(snapshot["model_tcp"]["euler_xyz_rad"], dtype=np.float64)
    quat = quat_wxyz_from_euler_xyz(euler)
    joints = np.asarray(snapshot["joint_degrees"], dtype=np.float64)
    status = snapshot["status"]
    print(
        f"{prefix} "
        f"joints_deg={np.round(joints, 3).tolist()} "
        f"position_m={np.round(position, 6).tolist()} "
        f"quat_wxyz={np.round(quat, 6).tolist()} "
        f"euler_xyz_rad={np.round(euler, 6).tolist()} "
        f"arm_status=0x{status['arm_status']:02X} "
        f"mode_feed=0x{status['mode_feed']:02X}",
        flush=True,
    )


def wait_for_state(robot: PiperRobot, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if robot.latest_state() is not None:
            return
        if robot.error is not None:
            raise RuntimeError("Piper feedback thread failed") from robot.error
        time.sleep(0.02)
    raise TimeoutError("Timed out waiting for Piper feedback")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move Piper to fixed joint angles, then continuously print model TCP pose."
    )
    parser.add_argument("--config", default="deploy/configs/piper_gemini_d435i.yaml")
    parser.add_argument("--target-deg", nargs=6, type=float, default=DEFAULT_TARGET_DEG.tolist())
    parser.add_argument("--speed-percent", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--tolerance-deg", type=float, default=1.5)
    parser.add_argument("--print-hz", type=float, default=5.0)
    parser.add_argument("--can-interface", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.speed_percent <= 10:
        raise ValueError("--speed-percent must be in [1, 10]")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    if args.tolerance_deg <= 0.0:
        raise ValueError("--tolerance-deg must be positive")
    if args.print_hz <= 0.0:
        raise ValueError("--print-hz must be positive")

    config = load_config(args.config)
    if args.can_interface is not None:
        config.robot.can_interface = args.can_interface
    target_deg = np.asarray(args.target_deg, dtype=np.float64)
    workspace_min = np.asarray(config.safety.workspace_min_m, dtype=np.float64)
    workspace_max = np.asarray(config.safety.workspace_max_m, dtype=np.float64)

    robot = PiperRobot(config.robot)
    motion_started = False
    try:
        robot.start()
        wait_for_state(robot)
        print("=== JOINT TARGET TCP PRINTER ===")
        print("CAN interface:", config.robot.can_interface)
        print("Target joints deg:", np.round(target_deg, 3).tolist())
        print("Speed percent:", args.speed_percent)
        print_tcp_snapshot(robot, "current")

        if not args.execute:
            print("DRY RUN ONLY: no enable or motion command was sent.")
            print("Re-run with --execute --confirm-motion to move, or Ctrl+C to stop this read-only connection.")
            while True:
                print_tcp_snapshot(robot, "current")
                time.sleep(1.0 / args.print_hz)

        if not args.confirm_motion:
            raise RuntimeError("Motion requires both --execute and --confirm-motion")
        print("Clear the swept volume and keep the emergency stop in hand.")
        typed = input(f"Type exactly {CONFIRM_TEXT} to move: ").strip()
        if typed != CONFIRM_TEXT:
            raise RuntimeError("Confirmation text did not match; motion cancelled")

        motion_started = True
        robot.enable_motion()
        print("Moving to joint target...")
        final_deg = robot.move_to_joint_target(
            target_deg=target_deg,
            speed_percent=args.speed_percent,
            timeout_s=args.timeout,
            workspace_min_m=workspace_min,
            workspace_max_m=workspace_max,
            close_gripper=False,
            tolerance_deg=args.tolerance_deg,
        )
        print("Reached/held joints deg:", np.round(final_deg, 3).tolist())
        print("Now continuously printing model TCP pose. Ctrl+C stops and holds current joints.")
        while True:
            print_tcp_snapshot(robot, "current")
            time.sleep(1.0 / args.print_hz)
    except KeyboardInterrupt:
        print("\nInterrupted; holding current joints if motion was enabled.")
    finally:
        robot.stop()
        if motion_started:
            print("Stopped Piper connection after sending hold-current-position.")


if __name__ == "__main__":
    main()
