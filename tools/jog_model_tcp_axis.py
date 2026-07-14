from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from deploy.config import load_config
from deploy.devices.piper_robot import PiperRobot
from deploy.kinematics import HostIKError


CONFIRM_TEXT = "JOG_MODEL_TCP_AXIS"
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


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


def wait_for_state(robot: PiperRobot, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if robot.latest_state() is not None:
            return
        if robot.error is not None:
            raise RuntimeError("Piper feedback thread failed") from robot.error
        time.sleep(0.02)
    raise TimeoutError("Timed out waiting for Piper feedback")


def write_event(handle, event: dict[str, object]) -> None:
    handle.write(json.dumps(jsonable(event), ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def summarize_ik_failure(error: HostIKError) -> None:
    diagnostics = error.diagnostics
    print("\n=== HOST IK REJECT ===")
    print(str(error))
    print("target_sdk_link6:", json.dumps(jsonable(diagnostics.get("target_sdk_link6")), ensure_ascii=False))
    print("current_joint_degrees:", np.round(diagnostics.get("current_joint_degrees", []), 3).tolist())
    print("position_tolerance_m:", diagnostics.get("position_tolerance_m"))
    print("rotation_tolerance_rad:", diagnostics.get("rotation_tolerance_rad"))
    print("max_joint_step_deg:", diagnostics.get("max_joint_step_deg"))
    print("pose_projection_enabled:", diagnostics.get("pose_projection_enabled"))
    for key in ("candidates", "projection_candidates"):
        candidates = diagnostics.get(key) or []
        print(f"{key}: {len(candidates)}")
        for index, candidate in enumerate(candidates[:5]):
            print(
                " ",
                index,
                "accepted=", candidate.get("accepted"),
                "reason=", candidate.get("rejection_reason"),
                "pos_err_m=", round(float(candidate.get("position_error_m", 0.0)), 6),
                "rot_err_rad=", round(float(candidate.get("rotation_error_rad", 0.0)), 6),
                "min_margin_deg=", round(float(candidate.get("minimum_limit_margin_deg", 0.0)), 3),
                "max_step_deg=", round(float(candidate.get("maximum_step_from_current_deg", 0.0)), 3),
                "joints=",
                np.round(candidate.get("joint_degrees", []), 3).tolist(),
            )


def pose_action_from_snapshot(
    snapshot: dict[str, object],
    delta_m: np.ndarray,
    gripper_m: float,
) -> np.ndarray:
    model_tcp = snapshot["model_tcp"]
    position = np.asarray(model_tcp["position_m"], dtype=np.float64) + delta_m
    euler = np.asarray(model_tcp["euler_xyz_rad"], dtype=np.float64)
    return np.concatenate([position, euler, [float(gripper_m)]])


def signed_projection(delta: np.ndarray, axis: str) -> float:
    return float(delta[AXIS_INDEX[axis]])


def fk_model_tcp_snapshot(robot: PiperRobot) -> dict[str, object] | None:
    piper = getattr(robot, "_piper", None)
    if piper is None:
        return None
    try:
        fk = np.asarray(piper.GetFK("feedback"), dtype=np.float64)
    except Exception as error:
        return {"error": str(error)}
    if fk.shape != (6, 6) or not np.isfinite(fk).all() or not np.any(fk):
        return {"error": f"invalid GetFK feedback shape/value: {fk.shape}"}
    sdk_pose = fk[-1]
    model_position, model_euler = robot.frames.sdk_to_model_pose(
        sdk_pose[:3] / 1000.0,
        np.radians(sdk_pose[3:]),
    )
    return {
        "sdk_link6_mm_deg": sdk_pose,
        "model_tcp": {
            "position_m": model_position,
            "euler_xyz_rad": model_euler,
        },
    }


def fk_position(snapshot: dict[str, object] | None) -> np.ndarray | None:
    if not snapshot or "model_tcp" not in snapshot:
        return None
    return np.asarray(snapshot["model_tcp"]["position_m"], dtype=np.float64)


def wait_for_fk(
    robot: PiperRobot,
    timeout_s: float = 8.0,
    allow_zero_start: bool = False,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last_error: object = None
    stable_count = 0
    previous_position: np.ndarray | None = None
    while time.monotonic() < deadline:
        feedback = robot.read_feedback_snapshot()
        joints = np.asarray(feedback["joint_degrees"], dtype=np.float64)
        fk_snapshot = fk_model_tcp_snapshot(robot)
        if not fk_snapshot or "model_tcp" not in fk_snapshot:
            last_error = fk_snapshot
            stable_count = 0
            previous_position = None
        else:
            position = np.asarray(
                fk_snapshot["model_tcp"]["position_m"], dtype=np.float64
            )
            near_zero_joints = float(np.max(np.abs(joints))) < 1.0
            plausible_position = position[0] > 0.12 and position[2] > 0.04
            if near_zero_joints and not allow_zero_start:
                last_error = {
                    "reason": "near-zero joint feedback",
                    "joint_degrees": joints,
                    "fk_position_m": position,
                }
                stable_count = 0
                previous_position = None
            elif not plausible_position and not allow_zero_start:
                last_error = {
                    "reason": "implausible FK model TCP",
                    "joint_degrees": joints,
                    "fk_position_m": position,
                }
                stable_count = 0
                previous_position = None
            else:
                if previous_position is not None and float(
                    np.linalg.norm(position - previous_position)
                ) <= 0.001:
                    stable_count += 1
                else:
                    stable_count = 1
                previous_position = position
                if stable_count >= 5:
                    return fk_snapshot
        if robot.error is not None:
            raise RuntimeError("Piper feedback thread failed") from robot.error
        time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for stable valid joint FK feedback: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Jog the DynamicVLA model TCP by a tiny axis-aligned delta and "
            "print whether Piper feedback moves in the commanded model-frame direction."
        )
    )
    parser.add_argument("--config", default="deploy/configs/piper_sequential.yaml")
    parser.add_argument("--axis", choices=tuple(AXIS_INDEX), default="x")
    parser.add_argument(
        "--state-source",
        choices=("endpose", "fk"),
        default="endpose",
        help="Use SDK EndPose feedback or joint FK as the start pose for the target.",
    )
    parser.add_argument("--delta-mm", type=float, default=5.0)
    parser.add_argument(
        "--allow-zero-start",
        action="store_true",
        help="Allow a near-zero six-joint feedback start pose. Disabled by default because stale Piper feedback often appears as all zeros.",
    )
    parser.add_argument("--start-timeout", type=float, default=8.0)
    parser.add_argument("--speed-percent", type=int, default=None)
    parser.add_argument("--sample-hz", type=float, default=25.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--gripper", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default=None,
        help="JSONL output path; default is deploy/runs/model-tcp-axis-<timestamp>.jsonl",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.2 <= abs(args.delta_mm) <= 10.0:
        raise ValueError("--delta-mm magnitude must be between 0.2 and 10.0")
    if args.sample_hz <= 0.0:
        raise ValueError("--sample-hz must be positive")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")

    config = load_config(args.config)
    if args.speed_percent is not None:
        if not 1 <= args.speed_percent <= 10:
            raise ValueError("--speed-percent must be in [1, 10]")
        config.robot.command_speed_percent = args.speed_percent

    output_path = (
        Path(args.output)
        if args.output
        else Path(config.runtime.output_dir)
        / ("model-tcp-axis-" + time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    delta_m = np.zeros(3, dtype=np.float64)
    delta_m[AXIS_INDEX[args.axis]] = args.delta_mm / 1000.0

    robot = PiperRobot(config.robot)
    motion_started = False
    with output_path.open("w", encoding="utf-8") as log:
        try:
            robot.start()
            raw_piper = getattr(robot, "_piper", None)
            if raw_piper is not None:
                try:
                    raw_piper.EnableFkCal()
                except Exception as error:
                    print("WARNING: EnableFkCal failed:", error)
            wait_for_state(robot)
            start = robot.read_feedback_snapshot()
            if args.state_source == "fk":
                start_fk = wait_for_fk(
                    robot,
                    timeout_s=args.start_timeout,
                    allow_zero_start=args.allow_zero_start,
                )
                source_snapshot = {
                    "model_tcp": start_fk["model_tcp"],
                }
            else:
                start_fk = fk_model_tcp_snapshot(robot)
                source_snapshot = start
            action = pose_action_from_snapshot(source_snapshot, delta_m, args.gripper)
            command = robot.describe_action(action)
            try:
                preview = robot.preview_action(action)
            except HostIKError as error:
                write_event(
                    log,
                    {
                        "event": "model_tcp_axis_preview_reject",
                        "config": args.config,
                        "axis": args.axis,
                        "delta_mm": args.delta_mm,
                        "state_source": args.state_source,
                        "start": start,
                        "start_fk": start_fk,
                        "target_action": action,
                        "planned_command": command,
                        "diagnostics": error.diagnostics,
                    },
                )
                print("=== MODEL TCP AXIS JOG DIAGNOSTIC ===")
                print("Output JSONL:", output_path)
                print("Control backend:", config.robot.control_backend)
                print("Commanded model axis:", args.axis.upper(), f"{args.delta_mm:+.3f} mm")
                print("Start model TCP position m:", np.round(start["model_tcp"]["position_m"], 6).tolist())
                print("Target model TCP position m:", np.round(action[:3], 6).tolist())
                print(
                    "Target SDK link6 position mm:",
                    np.round(command["sdk_link6"]["position_mm"], 3).tolist(),
                )
                summarize_ik_failure(error)
                print("\nNo motion command was sent. Try a smaller --delta-mm, e.g. 1.0.")
                return

            start_position = np.asarray(
                start["model_tcp"]["position_m"], dtype=np.float64
            )
            target_position = action[:3]

            write_event(
                log,
                {
                    "event": "model_tcp_axis_plan",
                    "config": args.config,
                    "control_backend": config.robot.control_backend,
                    "axis": args.axis,
                    "delta_mm": args.delta_mm,
                    "state_source": args.state_source,
                    "start": start,
                    "start_fk": start_fk,
                    "target_action": action,
                    "planned_command": command,
                    "preview": preview,
                    "execute": bool(args.execute),
                },
            )

            print("=== MODEL TCP AXIS JOG DIAGNOSTIC ===")
            print("Output JSONL:", output_path)
            print("Control backend:", config.robot.control_backend)
            print("Commanded model axis:", args.axis.upper(), f"{args.delta_mm:+.3f} mm")
            print("Target start pose source:", args.state_source)
            print("Configured speed:", config.robot.command_speed_percent, "%")
            print("Start model TCP position m (SDK EndPose):", np.round(start_position, 6).tolist())
            start_fk_position = fk_position(start_fk)
            if start_fk_position is not None:
                print("Start model TCP position m (joint FK):", np.round(start_fk_position, 6).tolist())
                print("EndPose - FK start delta mm:", np.round((start_position - start_fk_position) * 1000.0, 3).tolist())
            else:
                print("Start joint FK unavailable:", start_fk)
            if args.state_source == "fk" and start_fk_position is not None:
                print("Target is based on joint FK start + delta.")
            print("Target model TCP position m:", np.round(target_position, 6).tolist())
            print(
                "Target SDK link6 position mm:",
                np.round(command["sdk_link6"]["position_mm"], 3).tolist(),
            )
            print(
                "Target SDK link6 euler xyz deg:",
                np.round(command["sdk_link6"]["euler_xyz_deg"], 3).tolist(),
            )
            if config.robot.control_backend in {"host_ik_move_j", "host_pink_ik_move_j", "host_diff_ik_move_j"}:
                print(
                    "Preview current joints deg:",
                    np.round(preview["current_joint_degrees"], 3).tolist(),
                )
                print(
                    "Preview selected joints deg:",
                    np.round(preview["selected_joint_degrees"], 3).tolist(),
                )
                print(
                    "Preview max joint step deg:",
                    round(float(preview["requested_max_joint_step_deg"]), 4),
                )
                print("Preview pose_projected:", bool(preview["pose_projected"]))

            if not args.execute:
                print("\nDRY RUN ONLY: no enable or motion command was sent.")
                return
            if not args.confirm_motion:
                raise RuntimeError("Motion requires both --execute and --confirm-motion")

            print("\nClear the swept volume and keep the emergency stop in hand.")
            print("Run only one Piper process. This tool will hold current joints on exit.")
            typed = input(f"Type exactly {CONFIRM_TEXT} to move: ").strip()
            if typed != CONFIRM_TEXT:
                raise RuntimeError("Confirmation text did not match; motion cancelled")

            robot.enable_motion()
            motion_started = True

            period = 1.0 / args.sample_hz
            deadline = time.monotonic() + args.duration
            index = 0
            max_abs_projection_m = 0.0
            final_snapshot = start
            final_delta = np.zeros(3, dtype=np.float64)

            while time.monotonic() < deadline:
                if robot.error is not None:
                    raise RuntimeError("Piper feedback thread failed") from robot.error
                try:
                    prepared = robot.preview_action(action)
                except HostIKError as error:
                    write_event(
                        log,
                        {
                            "event": "model_tcp_axis_runtime_reject",
                            "index": index,
                            "target_action": action,
                            "diagnostics": error.diagnostics,
                        },
                    )
                    summarize_ik_failure(error)
                    break
                command_result = robot.command_action(action, prepared=prepared)
                snapshot = robot.read_feedback_snapshot()
                fk_snapshot = fk_model_tcp_snapshot(robot)
                position = np.asarray(
                    snapshot["model_tcp"]["position_m"], dtype=np.float64
                )
                measured_delta = position - start_position
                fk_pos = fk_position(fk_snapshot)
                if fk_pos is not None and start_fk_position is not None:
                    fk_delta = fk_pos - start_fk_position
                else:
                    fk_delta = None
                projection = signed_projection(measured_delta, args.axis)
                fk_projection = (
                    signed_projection(fk_delta, args.axis) if fk_delta is not None else None
                )
                max_abs_projection_m = max(max_abs_projection_m, abs(projection))
                final_snapshot = snapshot
                final_delta = measured_delta

                write_event(
                    log,
                    {
                        "event": "model_tcp_axis_sample",
                        "index": index,
                        "target_action": action,
                        "preview": prepared,
                        "command_result": command_result,
                        "robot_current": snapshot,
                        "robot_current_fk": fk_snapshot,
                        "measured_delta_m": measured_delta,
                        "measured_fk_delta_m": fk_delta,
                        "commanded_delta_m": delta_m,
                    },
                )
                print(
                    "sample=", index,
                    "endpose_delta_mm=",
                    np.round(measured_delta * 1000.0, 3).tolist(),
                    "fk_delta_mm=",
                    None if fk_delta is None else np.round(fk_delta * 1000.0, 3).tolist(),
                    "axis_projection_mm=",
                    round(projection * 1000.0, 3),
                    "fk_axis_projection_mm=",
                    None if fk_projection is None else round(fk_projection * 1000.0, 3),
                    "joints_deg=",
                    np.round(snapshot["joint_degrees"], 2).tolist(),
                    "status=",
                    f"0x{snapshot['status']['arm_status']:02X}",
                    "mode_feed=",
                    snapshot["status"]["mode_feed"],
                )
                if snapshot["status"]["arm_status"] != 0x00:
                    raise RuntimeError(
                        "Piper status became "
                        f"0x{snapshot['status']['arm_status']:02X}; stopping"
                    )
                index += 1
                sleep_s = period - (time.monotonic() % period)
                time.sleep(min(max(sleep_s, 0.0), period))

            final_position = np.asarray(
                final_snapshot["model_tcp"]["position_m"], dtype=np.float64
            )
            final_projection = signed_projection(final_delta, args.axis)
            same_direction = final_projection * delta_m[AXIS_INDEX[args.axis]] > 0.0
            write_event(
                log,
                {
                    "event": "model_tcp_axis_result",
                    "start_position_m": start_position,
                    "target_position_m": target_position,
                    "final_position_m": final_position,
                    "final_delta_m": final_delta,
                    "same_direction": same_direction,
                    "max_abs_axis_projection_m": max_abs_projection_m,
                },
            )
            print("\n=== RESULT ===")
            print("Final model TCP position m:", np.round(final_position, 6).tolist())
            print("Final measured delta mm:", np.round(final_delta * 1000.0, 3).tolist())
            print("Commanded axis projection mm:", round(args.delta_mm, 3))
            print("Measured axis projection mm:", round(final_projection * 1000.0, 3))
            print("Same sign as commanded axis:", same_direction)
            if max_abs_projection_m < 0.0005:
                print("WARNING: measured axis motion stayed below 0.5 mm.")
            elif not same_direction:
                print("FAIL: feedback moved opposite to the commanded model axis.")
            else:
                print("PASS: feedback moved in the commanded model-axis direction.")
        finally:
            raw_piper = getattr(robot, "_piper", None)
            if raw_piper is not None:
                try:
                    raw_piper.DisableFkCal()
                except Exception:
                    pass
            if motion_started:
                robot.stop()
            else:
                robot.stop()


if __name__ == "__main__":
    main()
