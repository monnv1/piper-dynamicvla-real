from __future__ import annotations

import argparse
import time

import numpy as np

from deploy.config import load_config
from deploy.devices.piper_robot import PiperRobot
from deploy.kinematics import HostIKError
from deploy.tools.jog_model_tcp_axis import wait_for_fk


CONFIRM_TEXT = "TELEOP_MODEL_TCP"
TRANSLATION_AXES = {"x": 0, "y": 1, "z": 2}
ROTATION_AXES = {"rx": 0, "ry": 1, "rz": 2}


def wait_for_state(robot: PiperRobot, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if robot.latest_state() is not None:
            return
        if robot.error is not None:
            raise RuntimeError("Piper feedback thread failed") from robot.error
        time.sleep(0.02)
    raise TimeoutError("Timed out waiting for Piper feedback")


def action_from_target(position: np.ndarray, euler: np.ndarray, gripper: float) -> np.ndarray:
    wrapped_euler = euler.copy()
    wrapped_euler[[0, 2]] %= 2.0 * np.pi
    return np.concatenate([position, wrapped_euler, [float(gripper)]])


def print_pose(label: str, position: np.ndarray, euler: np.ndarray) -> None:
    print(
        f"{label} pos_m={np.round(position, 5).tolist()} "
        f"euler_rad={np.round(euler, 4).tolist()} "
        f"euler_deg={np.round(np.degrees(euler), 2).tolist()}"
    )


def print_help() -> None:
    print("\nCommands:")
    print("  x+ x- y+ y- z+ z-        translate one step in model TCP frame axes")
    print("  rx+ rx- ry+ ry- rz+ rz-  rotate one step in Euler XYZ components")
    print("  step <mm>                set translation step, e.g. step 1")
    print("  rstep <deg>              set rotation step, e.g. rstep 2")
    print("  speed <1-10>             set command speed percent for subsequent commands")
    print("  show                     print current FK feedback and target")
    print("  sync                     reset target to current FK feedback")
    print("  hold                     command current FK joints/pose hold")
    print("  help                     print this help")
    print("  q                        quit and hold current joints\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive terminal jog control for DynamicVLA model TCP pose."
    )
    parser.add_argument("--config", default="deploy/configs/piper_model_tcp_axis_hostik_diagnostic.yaml")
    parser.add_argument("--step-mm", type=float, default=1.0)
    parser.add_argument("--rot-step-deg", type=float, default=2.0)
    parser.add_argument("--speed-percent", type=int, default=None)
    parser.add_argument("--gripper", type=float, default=0.0)
    parser.add_argument("--confirm-motion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.1 <= args.step_mm <= 10.0:
        raise ValueError("--step-mm must be in [0.1, 10.0]")
    if not 0.1 <= args.rot_step_deg <= 15.0:
        raise ValueError("--rot-step-deg must be in [0.1, 15.0]")

    config = load_config(args.config)
    config.robot.feedback_pose_source = "fk"
    config.robot.control_backend = "host_pink_ik_move_j"
    if args.speed_percent is not None:
        if not 1 <= args.speed_percent <= 10:
            raise ValueError("--speed-percent must be in [1, 10]")
        config.robot.command_speed_percent = args.speed_percent

    print("=== MODEL TCP TERMINAL TELEOP ===")
    print("Config:", args.config)
    print("Backend:", config.robot.control_backend)
    print("Feedback pose source: fk")
    print("Speed percent:", config.robot.command_speed_percent)
    print("Translation step mm:", args.step_mm)
    print("Rotation step deg:", args.rot_step_deg)
    print("This tool sends host IK MOVE J commands. Run only one Piper process.")
    if not args.confirm_motion:
        raise RuntimeError("Motion requires --confirm-motion")
    typed = input(f"Type exactly {CONFIRM_TEXT} to enable motion: ").strip()
    if typed != CONFIRM_TEXT:
        raise RuntimeError("Confirmation text did not match; cancelled")

    robot = PiperRobot(config.robot)
    motion_started = False
    try:
        robot.start()
        raw_piper = getattr(robot, "_piper", None)
        if raw_piper is not None:
            raw_piper.EnableFkCal()
        wait_for_state(robot)
        start_fk = wait_for_fk(robot, timeout_s=8.0, allow_zero_start=False)
        model_tcp = start_fk["model_tcp"]
        target_position = np.asarray(model_tcp["position_m"], dtype=np.float64)
        target_euler = np.asarray(model_tcp["euler_xyz_rad"], dtype=np.float64)

        robot.enable_motion()
        motion_started = True
        print_help()
        print_pose("Initial target", target_position, target_euler)

        step_m = args.step_mm / 1000.0
        rot_step_rad = np.radians(args.rot_step_deg)
        index = 0
        while True:
            command = input("tcp> ").strip().lower()
            if not command:
                continue
            parts = command.split()
            op = parts[0]
            if op in {"q", "quit", "exit"}:
                break
            if op == "help":
                print_help()
                continue
            if op == "step":
                if len(parts) != 2:
                    print("usage: step <mm>")
                    continue
                value = float(parts[1])
                if not 0.1 <= value <= 10.0:
                    print("step must be in [0.1, 10.0] mm")
                    continue
                step_m = value / 1000.0
                print("translation step mm:", value)
                continue
            if op == "rstep":
                if len(parts) != 2:
                    print("usage: rstep <deg>")
                    continue
                value = float(parts[1])
                if not 0.1 <= value <= 15.0:
                    print("rotation step must be in [0.1, 15.0] deg")
                    continue
                rot_step_rad = np.radians(value)
                print("rotation step deg:", value)
                continue
            if op == "speed":
                if len(parts) != 2:
                    print("usage: speed <1-10>")
                    continue
                value = int(parts[1])
                if not 1 <= value <= 10:
                    print("speed must be in [1, 10]")
                    continue
                config.robot.command_speed_percent = value
                robot.config.command_speed_percent = value
                print("speed percent:", value)
                continue
            if op == "show":
                fk = wait_for_fk(robot, timeout_s=3.0, allow_zero_start=False)
                current_position = np.asarray(fk["model_tcp"]["position_m"], dtype=np.float64)
                current_euler = np.asarray(fk["model_tcp"]["euler_xyz_rad"], dtype=np.float64)
                print_pose("Current FK", current_position, current_euler)
                print_pose("Target", target_position, target_euler)
                print("target-current mm:", np.round((target_position - current_position) * 1000.0, 3).tolist())
                continue
            if op == "sync":
                fk = wait_for_fk(robot, timeout_s=3.0, allow_zero_start=False)
                target_position = np.asarray(fk["model_tcp"]["position_m"], dtype=np.float64)
                target_euler = np.asarray(fk["model_tcp"]["euler_xyz_rad"], dtype=np.float64)
                print_pose("Synced target", target_position, target_euler)
                continue
            if op == "hold":
                fk = wait_for_fk(robot, timeout_s=3.0, allow_zero_start=False)
                target_position = np.asarray(fk["model_tcp"]["position_m"], dtype=np.float64)
                target_euler = np.asarray(fk["model_tcp"]["euler_xyz_rad"], dtype=np.float64)
            elif op in {axis + sign for axis in TRANSLATION_AXES for sign in ["+", "-"]}:
                axis = op[:-1]
                sign = 1.0 if op[-1] == "+" else -1.0
                target_position[TRANSLATION_AXES[axis]] += sign * step_m
            elif op in {axis + sign for axis in ROTATION_AXES for sign in ["+", "-"]}:
                axis = op[:-1]
                sign = 1.0 if op[-1] == "+" else -1.0
                target_euler[ROTATION_AXES[axis]] += sign * rot_step_rad
                target_euler[[0, 2]] %= 2.0 * np.pi
            else:
                print("unknown command; type 'help'")
                continue

            action = action_from_target(target_position, target_euler, args.gripper)
            try:
                preview = robot.preview_action(action)
                result = robot.command_action(action, prepared=preview)
            except HostIKError as error:
                print("HOST IK REJECT:", error)
                print("diagnostics selected keys:", {k: error.diagnostics.get(k) for k in ["pose_projected", "pose_projection_reason"]})
                continue

            time.sleep(0.05)
            fk = wait_for_fk(robot, timeout_s=3.0, allow_zero_start=False)
            current_position = np.asarray(fk["model_tcp"]["position_m"], dtype=np.float64)
            current_euler = np.asarray(fk["model_tcp"]["euler_xyz_rad"], dtype=np.float64)
            print(
                f"cmd={index} op={op} "
                f"current_pos={np.round(current_position, 5).tolist()} "
                f"target_pos={np.round(target_position, 5).tolist()} "
                f"err_mm={np.round((target_position - current_position) * 1000.0, 3).tolist()} "
                f"joints={np.round(result.get('selected_joint_degrees', []), 2).tolist()}"
            )
            index += 1
    finally:
        raw_piper = getattr(robot, "_piper", None)
        if raw_piper is not None:
            try:
                raw_piper.DisableFkCal()
            except Exception:
                pass
        if motion_started:
            print("Stopping: holding measured current joints and disconnecting.")
        robot.stop()


if __name__ == "__main__":
    main()
