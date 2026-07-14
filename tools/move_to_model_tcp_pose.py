from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from deploy.config import load_config
from deploy.devices.piper_robot import PiperRobot


DEFAULT_MODEL_POSITION_M = [0.373, 0.0, 0.271]
DEFAULT_MODEL_QUAT_WXYZ = [0.0, 0.9739, 0.0, 0.227]
CONFIRM_TEXT = "MOVE_PIPER_TO_MODEL_TCP"


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def quat_wxyz_to_euler_xyz(quat_wxyz: list[float]) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    quat = quat / norm
    xyzw = np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
    euler = Rotation.from_quat(xyzw).as_euler("xyz")
    euler[[0, 2]] %= 2.0 * np.pi
    return euler


def rotation_error_rad(current_euler: np.ndarray, target_euler: np.ndarray) -> float:
    current = Rotation.from_euler("xyz", current_euler)
    target = Rotation.from_euler("xyz", target_euler)
    return float((current.inv() * target).magnitude())


def pose_error(snapshot: dict[str, object], target_position: np.ndarray, target_euler: np.ndarray) -> dict[str, float]:
    model_tcp = snapshot["model_tcp"]
    current_position = np.asarray(model_tcp["position_m"], dtype=np.float64)
    current_euler = np.asarray(model_tcp["euler_xyz_rad"], dtype=np.float64)
    position_error = target_position - current_position
    return {
        "position_norm_m": float(np.linalg.norm(position_error)),
        "position_xyz_m": position_error.tolist(),
        "rotation_rad": rotation_error_rad(current_euler, target_euler),
    }


def wait_for_state(robot: PiperRobot, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if robot.latest_state() is not None:
            return
        if robot.error is not None:
            raise RuntimeError("Piper feedback thread failed") from robot.error
        time.sleep(0.02)
    raise TimeoutError("Timed out waiting for Piper feedback")


def validate_workspace(position_m: np.ndarray, workspace_min: np.ndarray, workspace_max: np.ndarray) -> None:
    invalid = np.flatnonzero((position_m < workspace_min) | (position_m > workspace_max))
    if invalid.size:
        details = ", ".join(
            f"axis {int(index)}={position_m[index]:.4f} not in "
            f"[{workspace_min[index]:.4f}, {workspace_max[index]:.4f}]"
            for index in invalid
        )
        raise RuntimeError(f"Target model TCP is outside configured workspace: {details}")


def write_event(handle, event: dict[str, object]) -> None:
    handle.write(json.dumps(jsonable(event), ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move Piper to a specified DynamicVLA model-TCP pose and log both "
            "model-TCP feedback and raw Piper SDK link6 feedback."
        )
    )
    parser.add_argument("--config", default="deploy/configs/piper_gemini_d435i.yaml")
    parser.add_argument("--position", nargs=3, type=float, default=DEFAULT_MODEL_POSITION_M)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quat-wxyz", nargs=4, type=float, default=DEFAULT_MODEL_QUAT_WXYZ)
    group.add_argument("--euler-xyz", nargs=3, type=float, default=None)
    parser.add_argument("--gripper", type=float, default=0.0)
    parser.add_argument("--can-interface", default=None)
    parser.add_argument("--speed-percent", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sample-hz", type=float, default=25.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.004)
    parser.add_argument("--rotation-tolerance-rad", type=float, default=0.06)
    parser.add_argument("--settle-samples", type=int, default=10)
    parser.add_argument("--output", default=None, help="JSONL output path; default is deploy/runs/model-tcp-<timestamp>.jsonl")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    if args.sample_hz <= 0.0:
        raise ValueError("--sample-hz must be positive")
    if args.position_tolerance_m <= 0.0 or args.rotation_tolerance_rad <= 0.0:
        raise ValueError("Pose tolerances must be positive")
    if args.settle_samples <= 0:
        raise ValueError("--settle-samples must be positive")

    config = load_config(args.config)
    if args.can_interface is not None:
        config.robot.can_interface = args.can_interface
    if args.speed_percent is not None:
        if not 1 <= args.speed_percent <= 10:
            raise ValueError("--speed-percent must be in [1, 10]")
        config.robot.command_speed_percent = args.speed_percent

    target_position = np.asarray(args.position, dtype=np.float64)
    if target_position.shape != (3,) or not np.isfinite(target_position).all():
        raise ValueError("--position must contain three finite values")
    if args.euler_xyz is None:
        target_euler = quat_wxyz_to_euler_xyz(args.quat_wxyz)
    else:
        target_euler = np.asarray(args.euler_xyz, dtype=np.float64)
        if target_euler.shape != (3,) or not np.isfinite(target_euler).all():
            raise ValueError("--euler-xyz must contain three finite values")
        target_euler[[0, 2]] %= 2.0 * np.pi

    workspace_min = np.asarray(config.safety.workspace_min_m, dtype=np.float64)
    workspace_max = np.asarray(config.safety.workspace_max_m, dtype=np.float64)
    if config.safety.enforce_workspace:
        validate_workspace(target_position, workspace_min, workspace_max)

    output_path = Path(args.output) if args.output else Path(config.runtime.output_dir) / (
        "model-tcp-" + time.strftime("%Y%m%d-%H%M%S") + ".jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    action = np.concatenate([target_position, target_euler, [float(args.gripper)]])
    robot = PiperRobot(config.robot)
    motion_started = False
    log_handle = output_path.open("w", encoding="utf-8")
    try:
        robot.start()
        wait_for_state(robot)
        current = robot.read_feedback_snapshot()
        command_description = robot.describe_action(action)
        preview = robot.preview_action(action)
        initial_error = pose_error(current, target_position, target_euler)
        header = {
            "event": "model_tcp_move_plan",
            "config": args.config,
            "can_interface": config.robot.can_interface,
            "control_backend": config.robot.control_backend,
            "target": command_description,
            "target_quat_wxyz": args.quat_wxyz if args.euler_xyz is None else None,
            "target_euler_xyz_rad": target_euler,
            "current": current,
            "initial_error": initial_error,
            "preview": preview,
            "tolerances": {
                "position_m": args.position_tolerance_m,
                "rotation_rad": args.rotation_tolerance_rad,
                "settle_samples": args.settle_samples,
            },
            "configured_sdk_to_model": {
                "translation_m": config.robot.sdk_to_model_translation_m,
                "euler_xyz_rad": config.robot.sdk_to_model_euler_xyz_rad,
            },
            "execute": bool(args.execute),
        }
        write_event(log_handle, header)

        print("=== MODEL TCP MOVE PLAN ===")
        print("Output JSONL:", output_path)
        print("Control backend:", config.robot.control_backend)
        print("Target model TCP position m:", np.round(target_position, 6).tolist())
        print("Target model TCP euler xyz rad:", np.round(target_euler, 6).tolist())
        print("Target SDK link6 position mm:", np.round(command_description["sdk_link6"]["position_mm"], 3).tolist())
        print("Target SDK link6 euler xyz deg:", np.round(command_description["sdk_link6"]["euler_xyz_deg"], 3).tolist())
        print("Current model TCP position m:", np.round(current["model_tcp"]["position_m"], 6).tolist())
        print("Current model TCP euler xyz rad:", np.round(current["model_tcp"]["euler_xyz_rad"], 6).tolist())
        print("Initial position error m:", round(initial_error["position_norm_m"], 6))
        print("Initial rotation error rad:", round(initial_error["rotation_rad"], 6))
        if config.robot.control_backend == "host_ik_move_j":
            print("Preview selected joints deg:", np.round(preview["selected_joint_degrees"], 3).tolist())

        if not args.execute:
            print("\nDRY RUN ONLY: connected and computed the target, but did not enable or move.")
            return
        if not args.confirm_motion:
            raise RuntimeError("Motion requires both --execute and --confirm-motion")

        print("\nBefore confirming, clear the swept volume and keep the emergency stop in hand.")
        print("Do not run deploy.run or any other Piper process concurrently.")
        typed = input(f"Type exactly {CONFIRM_TEXT} to move: ").strip()
        if typed != CONFIRM_TEXT:
            raise RuntimeError("Confirmation text did not match; motion cancelled")

        motion_started = True
        robot.enable_motion()
        period = 1.0 / args.sample_hz
        deadline = time.monotonic() + args.timeout
        settled = 0
        command_index = 0
        best_position_error = float("inf")
        best_error_time = time.monotonic()

        while time.monotonic() < deadline:
            if robot.error is not None:
                raise RuntimeError("Piper feedback thread failed") from robot.error
            prepared = robot.preview_action(action)
            command_result = robot.command_action(action, prepared=prepared)
            snapshot = robot.read_feedback_snapshot()
            error = pose_error(snapshot, target_position, target_euler)
            event = {
                "event": "model_tcp_move_sample",
                "index": command_index,
                "target": command_description,
                "preview": prepared,
                "command_result": command_result,
                "robot_current": snapshot,
                "error": error,
            }
            write_event(log_handle, event)

            print(
                "sample=", command_index,
                "pos_err_m=", round(error["position_norm_m"], 5),
                "rot_err_rad=", round(error["rotation_rad"], 4),
                "joints_deg=", np.round(snapshot["joint_degrees"], 2).tolist(),
                "status=", f"0x{snapshot['status']['arm_status']:02X}",
            )

            if snapshot["status"]["arm_status"] != 0x00:
                raise RuntimeError(
                    f"Piper status became 0x{snapshot['status']['arm_status']:02X}; stopping command stream"
                )
            if error["position_norm_m"] < best_position_error - 0.001:
                best_position_error = error["position_norm_m"]
                best_error_time = time.monotonic()
            elif time.monotonic() - best_error_time > 6.0 and error["position_norm_m"] > args.position_tolerance_m:
                raise RuntimeError("Model TCP feedback is not converging for 6s")

            if (
                error["position_norm_m"] <= args.position_tolerance_m
                and error["rotation_rad"] <= args.rotation_tolerance_rad
            ):
                settled += 1
                if settled >= args.settle_samples:
                    write_event(
                        log_handle,
                        {
                            "event": "model_tcp_move_reached",
                            "index": command_index,
                            "robot_current": snapshot,
                            "error": error,
                        },
                    )
                    print("Target reached within tolerance.")
                    return
            else:
                settled = 0
            command_index += 1
            time.sleep(period)

        raise TimeoutError(f"Timed out after {args.timeout:.1f}s before reaching target")
    except KeyboardInterrupt:
        print("\nInterrupted; robot.stop() will hold the measured current joints if motion started.")
    finally:
        if motion_started:
            final_snapshot = None
            try:
                final_snapshot = robot.read_feedback_snapshot()
            except Exception:
                pass
            write_event(
                log_handle,
                {
                    "event": "model_tcp_move_stop",
                    "motion_started": motion_started,
                    "final_before_hold": final_snapshot,
                },
            )
        robot.stop()
        log_handle.close()


if __name__ == "__main__":
    main()
