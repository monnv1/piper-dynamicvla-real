from __future__ import annotations

import argparse
import time

import numpy as np

from deploy.config import load_config
from deploy.devices.piper_frames import PiperFrameTransform
from deploy.tools.move_to_training_start import (
    JOINT_LIMITS_DEG,
    enable_with_timeout,
    read_joint_degrees,
    require_normal_status,
    resume_position_control,
    sample_model_tcp_path,
    send_joint_target,
    hold_current_position,
    validate_joint_values,
    validate_tcp_path,
)


CONFIRM_TEXT = "JOG_ONE_PIPER_JOINT"


def make_jog_target(
    current_deg: np.ndarray, joint_number: int, delta_deg: float
) -> np.ndarray:
    target = np.asarray(current_deg, dtype=np.float64).copy()
    target[joint_number - 1] += delta_deg
    return target


def validate_jog_target(target_deg: np.ndarray, joint_number: int) -> None:
    # Other joints may legitimately sit exactly on a hard limit (the current
    # Piper pose has J2=0 and J3=0). Apply the 0.2-degree soft margin only to
    # the joint being moved, while still hard-limit checking all six joints.
    validate_joint_values(target_deg, "jog target")
    index = joint_number - 1
    lower = JOINT_LIMITS_DEG[index, 0] + 0.2
    upper = JOINT_LIMITS_DEG[index, 1] - 0.2
    if not lower <= target_deg[index] <= upper:
        raise RuntimeError(
            f"jog target J{joint_number}={target_deg[index]:.3f} not in "
            f"soft range [{lower:.1f}, {upper:.1f}]"
        )


def sample_between_joint_targets(
    start_deg: np.ndarray,
    target_deg: np.ndarray,
    dh_is_offset: int,
    frames: PiperFrameTransform,
    samples: int = 21,
) -> np.ndarray:
    from piper_sdk import C_PiperForwardKinematics

    fk = C_PiperForwardKinematics(dh_is_offset)
    start_rad = np.radians(start_deg)
    target_rad = np.radians(target_deg)
    positions = []
    for fraction in np.linspace(0.0, 1.0, samples):
        joints = start_rad + (target_rad - start_rad) * fraction
        sdk_pose = np.asarray(fk.CalFK(joints)[-1], dtype=np.float64)
        model_position, _ = frames.sdk_to_model_pose(
            sdk_pose[:3] / 1000.0,
            np.radians(sdk_pose[3:]),
        )
        positions.append(model_position)
    return np.stack(positions)


def move_and_wait(
    piper,
    target_deg: np.ndarray,
    speed_percent: int,
    tolerance_deg: float,
    timeout_s: float,
    label: str,
) -> None:
    deadline = time.monotonic() + timeout_s
    settled = 0
    last_print = 0.0
    while time.monotonic() < deadline:
        require_normal_status(piper, label)
        send_joint_target(piper, target_deg, speed_percent)
        current_deg, _ = read_joint_degrees(piper)
        validate_joint_values(current_deg, "feedback joints")
        max_error = float(np.max(np.abs(target_deg - current_deg)))
        now = time.monotonic()
        if now - last_print >= 0.25:
            print(
                f"{label}: joints_deg={np.round(current_deg, 3).tolist()} "
                f"max_error_deg={max_error:.3f}"
            )
            last_print = now
        if max_error <= tolerance_deg:
            settled += 1
            if settled >= 5:
                return
        else:
            settled = 0
        time.sleep(0.05)
    raise TimeoutError(f"{label} did not settle within {timeout_s:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slowly jog one Piper joint and hold the final angle"
    )
    parser.add_argument("--joint", type=int, required=True, choices=range(1, 7))
    parser.add_argument("--delta-deg", type=float, default=1.0)
    parser.add_argument("--speed-percent", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--tolerance-deg", type=float, default=0.2)
    parser.add_argument(
        "--config", default="deploy/configs/piper_gemini_d435i.yaml"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true")
    args = parser.parse_args()

    if not 0.2 <= abs(args.delta_deg) <= 40.0:
        raise ValueError("Absolute --delta-deg must be between 0.2 and 40.0")
    if not 1 <= args.speed_percent <= 10:
        raise ValueError("--speed-percent is hard limited to 1..10")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if not 0.05 <= args.tolerance_deg <= 0.5:
        raise ValueError("--tolerance-deg must be between 0.05 and 0.5")

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

    cleanup_required = False
    piper.ConnectPort(piper_init=False, start_thread=True)
    try:
        time.sleep(1.0)
        start_deg, feedback_hz = read_joint_degrees(piper)
        if feedback_hz <= 0:
            raise RuntimeError("No Piper joint feedback")
        require_normal_status(piper, "preflight")
        validate_joint_values(start_deg, "captured start joints")
        target_deg = make_jog_target(start_deg, args.joint, args.delta_deg)
        validate_jog_target(target_deg, args.joint)

        tcp_path = sample_between_joint_targets(
            start_deg,
            target_deg,
            config.robot.dh_is_offset,
            frames,
        )
        validate_tcp_path(
            tcp_path,
            np.asarray(config.safety.workspace_min_m),
            np.asarray(config.safety.workspace_max_m),
        )
        tcp_delta_mm = (tcp_path[-1] - tcp_path[0]) * 1000.0

        print("\n=== SINGLE-JOINT JOG PLAN ===")
        print("Joint feedback Hz:", feedback_hz)
        print("Joint:", args.joint)
        print("Delta:", args.delta_deg, "deg")
        print("Speed:", args.speed_percent, "%")
        print("Captured start joints (deg):", np.round(start_deg, 3).tolist())
        print("Jog target joints     (deg):", np.round(target_deg, 3).tolist())
        print("Approx TCP delta XYZ  (mm):", np.round(tcp_delta_mm, 3).tolist())
        print("The tool will remain at the jog target and hold it with motors enabled.")
        print("Gripper: unchanged")

        if not args.execute:
            print("\nDRY RUN ONLY: no motor was enabled and no command was sent.")
            return
        if not args.confirm_motion:
            raise RuntimeError("Motion requires both --execute and --confirm-motion")

        print("\nClear the swept volume and keep the physical emergency stop in hand.")
        print("Run only one Piper process. Watch the selected joint, camera and cables.")
        typed = input(f"Type exactly {CONFIRM_TEXT} to jog: ").strip()
        if typed != CONFIRM_TEXT:
            raise RuntimeError("Confirmation text did not match; motion cancelled")

        cleanup_required = True
        enable_with_timeout(piper, timeout_s=5.0)
        resumed = resume_position_control(piper)
        print("Position control active at:", np.round(resumed, 3).tolist())
        move_and_wait(
            piper,
            target_deg,
            args.speed_percent,
            args.tolerance_deg,
            args.timeout,
            "jog",
        )
        print("Jog target reached; holding the final joint angles.")
    except KeyboardInterrupt:
        print("\nInterrupted; terminating motion and holding the current position.")
    finally:
        if cleanup_required:
            held = hold_current_position(piper)
            print("Holding joints with motors enabled:", np.round(held, 3).tolist())
            print("WARNING: support the arm before any later DisableArm/power-off.")
        piper.DisconnectPort()


if __name__ == "__main__":
    main()
