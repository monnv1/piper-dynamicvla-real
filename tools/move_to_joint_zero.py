from __future__ import annotations

import argparse
import time

import numpy as np

from deploy.config import load_config
from deploy.devices.piper_frames import PiperFrameTransform
from deploy.tools.jog_piper_joint import move_and_wait, sample_between_joint_targets
from deploy.tools.move_to_training_start import (
    enable_with_timeout,
    hold_current_position,
    read_joint_degrees,
    require_normal_status,
    resume_position_control,
    validate_joint_values,
    validate_tcp_path,
)


JOINT_ZERO_DEG = np.zeros(6, dtype=np.float64)
CONFIRM_TEXT = "MOVE_PIPER_SLOWLY_TO_CALIBRATED_ZERO"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slowly MOVE J to the existing calibrated six-joint zero"
    )
    parser.add_argument(
        "--config", default="deploy/configs/piper_gemini_d435i.yaml"
    )
    parser.add_argument("--speed-percent", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--tolerance-deg", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.speed_percent <= 10:
        raise ValueError("--speed-percent is hard limited to 1..10")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if not 0.2 <= args.tolerance_deg <= 2.0:
        raise ValueError("--tolerance-deg must be between 0.2 and 2.0")

    config = load_config(args.config)
    frames = PiperFrameTransform(
        config.robot.sdk_to_model_translation_m,
        config.robot.sdk_to_model_euler_xyz_rad,
    )
    validate_joint_values(JOINT_ZERO_DEG, "calibrated joint zero")

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
        validate_joint_values(start_deg, "current joints")

        tcp_path = sample_between_joint_targets(
            start_deg,
            JOINT_ZERO_DEG,
            config.robot.dh_is_offset,
            frames,
            samples=51,
        )
        validate_tcp_path(
            tcp_path,
            np.asarray(config.safety.workspace_min_m),
            np.asarray(config.safety.workspace_max_m),
        )

        print("\n=== CALIBRATED JOINT-ZERO PLAN ===")
        print("Current joints (deg):", np.round(start_deg, 3).tolist())
        print("Target joints  (deg):", JOINT_ZERO_DEG.tolist())
        print("Joint deltas   (deg):", np.round(-start_deg, 3).tolist())
        print("Speed:", args.speed_percent, "%")
        print("Approx model-TCP path min (m):", np.round(tcp_path.min(0), 4).tolist())
        print("Approx model-TCP path max (m):", np.round(tcp_path.max(0), 4).tolist())
        print("Model TCP at joint zero (m):", np.round(tcp_path[-1], 4).tolist())
        print("Gripper: unchanged")
        print("This moves to the existing zero; it does NOT recalibrate encoder zero.")
        print("Joint zero is NOT the DynamicVLA training start configuration.")

        if not args.execute:
            print("\nDRY RUN ONLY: no motor was enabled and no command was sent.")
            return
        if not args.confirm_motion:
            raise RuntimeError("Motion requires both --execute and --confirm-motion")

        print("\nClear the complete swept volume and keep physical emergency stop ready.")
        typed = input(f"Type exactly {CONFIRM_TEXT} to move: ").strip()
        if typed != CONFIRM_TEXT:
            raise RuntimeError("Confirmation text did not match; motion cancelled")

        cleanup_required = True
        enable_with_timeout(piper, timeout_s=5.0)
        resumed = resume_position_control(piper)
        print("Position control active at:", np.round(resumed, 3).tolist())
        move_and_wait(
            piper,
            JOINT_ZERO_DEG,
            args.speed_percent,
            args.tolerance_deg,
            args.timeout,
            "go-zero",
        )
        print("Calibrated joint zero reached; holding with motors enabled.")
    except KeyboardInterrupt:
        print("\nInterrupted; holding the current position.")
    finally:
        if cleanup_required:
            held = hold_current_position(piper)
            print("Holding joints with motors enabled:", np.round(held, 3).tolist())
            print("WARNING: support the arm before any later DisableArm/power-off.")
        piper.DisconnectPort()


if __name__ == "__main__":
    main()
