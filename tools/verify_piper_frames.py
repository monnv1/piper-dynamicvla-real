from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

from deploy.config import load_config
from deploy.devices.piper_frames import PiperFrameTransform


STATUS_NAMES = {
    0x00: "normal",
    0x01: "emergency_stop",
    0x02: "no_ik_solution",
    0x03: "singularity",
    0x04: "target_joint_limit",
    0x05: "joint_communication_error",
    0x06: "joint_brake_closed",
    0x07: "collision",
    0x08: "teach_overspeed",
    0x09: "joint_error",
    0x0A: "other_error",
}


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def rotation_error_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> float:
    a = Rotation.from_euler("xyz", a_deg, degrees=True)
    b = Rotation.from_euler("xyz", b_deg, degrees=True)
    return float(np.degrees((a.inv() * b).magnitude()))


def status_code(piper) -> int:
    wrapper = piper.GetArmStatus()
    status = getattr(wrapper, "arm_status", wrapper)
    return int(getattr(status, "arm_status", -1))


def joint_degrees(piper) -> list[float]:
    joint = piper.GetArmJointMsgs().joint_state
    return [
        float(joint.joint_1) / 1000.0,
        float(joint.joint_2) / 1000.0,
        float(joint.joint_3) / 1000.0,
        float(joint.joint_4) / 1000.0,
        float(joint.joint_5) / 1000.0,
        float(joint.joint_6) / 1000.0,
    ]


def end_pose_mm_deg(piper) -> np.ndarray:
    pose = piper.GetArmEndPoseMsgs().end_pose
    return np.asarray(
        [
            pose.X_axis / 1000.0,
            pose.Y_axis / 1000.0,
            pose.Z_axis / 1000.0,
            pose.RX_axis / 1000.0,
            pose.RY_axis / 1000.0,
            pose.RZ_axis / 1000.0,
        ],
        dtype=np.float64,
    )


def fk6_pose(piper) -> np.ndarray:
    fk = np.asarray(piper.GetFK("feedback"), dtype=np.float64)
    if fk.shape != (6, 6):
        raise RuntimeError(f"Unexpected GetFK shape: {fk.shape}")
    return fk[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only comparison of Piper EndPose, FK joint6 and model TCP"
    )
    parser.add_argument(
        "--config", default="deploy/configs/piper_gemini_d435i.yaml"
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0 or args.sample_hz <= 0:
        raise ValueError("--seconds and --sample-hz must be positive")

    config = load_config(args.config)
    frames = PiperFrameTransform(
        config.robot.sdk_to_model_translation_m,
        config.robot.sdk_to_model_euler_xyz_rad,
    )

    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(
        can_name=config.robot.can_interface,
        judge_flag=config.robot.official_can_adapter,
        can_auto_init=True,
        dh_is_offset=config.robot.dh_is_offset,
        start_sdk_joint_limit=True,
        start_sdk_gripper_limit=True,
    )

    print("READ-ONLY: this script never enables the arm or sends commands.")
    piper.ConnectPort(piper_init=False, start_thread=True)
    piper.EnableFkCal()
    end_samples: list[np.ndarray] = []
    fk_samples: list[np.ndarray] = []
    position_errors: list[float] = []
    rotation_errors: list[float] = []
    statuses: Counter[int] = Counter()
    last_joints: list[float] = []

    try:
        # Allow CAN feedback and the SDK FK thread to populate.
        time.sleep(1.0)
        deadline = time.monotonic() + args.seconds
        period = 1.0 / args.sample_hz
        while time.monotonic() < deadline:
            end = end_pose_mm_deg(piper)
            fk6 = fk6_pose(piper)
            code = status_code(piper)
            last_joints = joint_degrees(piper)

            if np.isfinite(end).all() and np.isfinite(fk6).all() and np.any(fk6):
                end_samples.append(end)
                fk_samples.append(fk6)
                position_errors.append(float(np.linalg.norm(end[:3] - fk6[:3])))
                rotation_errors.append(rotation_error_deg(end[3:], fk6[3:]))
            statuses[code] += 1
            time.sleep(period)
    finally:
        piper.DisableFkCal()
        piper.DisconnectPort()

    if not end_samples:
        raise RuntimeError("No valid FK samples; check CAN feedback and EnableFkCal support")

    end_array = np.stack(end_samples)
    fk_array = np.stack(fk_samples)
    end_median = np.median(end_array, axis=0)
    fk_median = np.median(fk_array, axis=0)
    pos_error = np.asarray(position_errors)
    rot_error = np.asarray(rotation_errors)

    model_position, model_euler = frames.sdk_to_model_pose(
        end_median[:3] / 1000.0,
        np.radians(end_median[3:]),
    )
    position_match = percentile(pos_error, 95) <= 5.0
    rotation_match = percentile(rot_error, 95) <= 2.0
    result = {
        "read_only": True,
        "samples": len(end_samples),
        "can_interface": config.robot.can_interface,
        "dh_is_offset": config.robot.dh_is_offset,
        "joint_degrees_last": last_joints,
        "end_pose_median_mm_deg": end_median.tolist(),
        "fk_joint6_median_mm_deg": fk_median.tolist(),
        "end_vs_fk_position_error_mm": {
            "median": percentile(pos_error, 50),
            "p95": percentile(pos_error, 95),
            "max": float(np.max(pos_error)),
        },
        "end_vs_fk_rotation_error_deg": {
            "median": percentile(rot_error, 50),
            "p95": percentile(rot_error, 95),
            "max": float(np.max(rot_error)),
        },
        "end_pose_matches_fk_joint6": bool(position_match and rotation_match),
        "match_thresholds": {"position_mm": 5.0, "rotation_deg": 2.0},
        "arm_status_counts": {
            f"0x{code:02X}_{STATUS_NAMES.get(code, 'unknown')}": count
            for code, count in sorted(statuses.items())
        },
        "configured_sdk_to_model": {
            "translation_m": config.robot.sdk_to_model_translation_m,
            "euler_xyz_rad": config.robot.sdk_to_model_euler_xyz_rad,
        },
        "model_tcp_from_end_pose_median": {
            "position_m": model_position.tolist(),
            "euler_xyz_rad": model_euler.tolist(),
        },
        "safe_to_keep_current_frame_transform": bool(
            position_match and rotation_match
        ),
    }
    print("\n=== COPY EVERYTHING BELOW ===")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
