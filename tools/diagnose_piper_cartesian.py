from __future__ import annotations

import argparse
import time

import numpy as np

from deploy.config import load_config
from deploy.tools.move_to_training_start import (
    arm_status_code,
    enable_with_timeout,
    hold_current_position,
)


CONFIRM_TEXT = "TEST_PIPER_CARTESIAN_1MM"
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


def read_pose_units(piper) -> np.ndarray:
    pose = piper.GetArmEndPoseMsgs().end_pose
    return np.asarray(
        [
            pose.X_axis,
            pose.Y_axis,
            pose.Z_axis,
            pose.RX_axis,
            pose.RY_axis,
            pose.RZ_axis,
        ],
        dtype=np.int64,
    )


def print_status(piper, label: str) -> None:
    wrapper = piper.GetArmStatus()
    status = getattr(wrapper, "arm_status", wrapper)
    code = int(getattr(status, "arm_status", -1))
    code_text = STATUS_NAMES.get(code, "unknown")
    print(f"\n[{label}]")
    print("motor_enable:", [bool(value) for value in piper.GetArmEnableStatus()])
    print("feedback_hz:", float(getattr(wrapper, "Hz", 0.0)))
    print("ctrl_mode:", f"0x{int(getattr(status, 'ctrl_mode', -1)):02X}")
    print("arm_status:", f"0x{code:02X} ({code_text})")
    print("mode_feed:", getattr(status, "mode_feed", "not_exposed"))
    print("motion_status:", getattr(status, "motion_status", "not_exposed"))


def print_mode_command(piper) -> None:
    wrapper = piper.GetArmCtrlCode151()
    command = getattr(wrapper, "ctrl_151", wrapper)
    print("\n[0x151 command echo]")
    print("echo_hz:", float(getattr(wrapper, "Hz", 0.0)))
    print("ctrl_mode:", f"0x{int(getattr(command, 'ctrl_mode', -1)):02X}")
    print("move_mode:", f"0x{int(getattr(command, 'move_mode', -1)):02X}")
    print("speed_percent:", int(getattr(command, "move_spd_rate_ctrl", -1)))
    print("mit_mode:", f"0x{int(getattr(command, 'mit_mode', -1)):02X}")


def wait_for_feedback(piper, start: np.ndarray, timeout_s: float) -> tuple[np.ndarray, float]:
    deadline = time.monotonic() + timeout_s
    latest = start.copy()
    maximum_mm = 0.0
    while time.monotonic() < deadline:
        latest = read_pose_units(piper)
        displacement_mm = float(np.linalg.norm(latest[:3] - start[:3]) / 1000.0)
        maximum_mm = max(maximum_mm, displacement_mm)
        status = arm_status_code(piper)
        if status != 0x00 or displacement_mm >= 0.20:
            break
        time.sleep(0.02)
    return latest, maximum_mm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Piper MOVE P / EndPoseCtrl with a tiny Cartesian jog."
    )
    parser.add_argument(
        "--config",
        default="deploy/configs/piper_gemini_d435i.yaml",
    )
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--delta-mm", type=float, default=1.0)
    parser.add_argument("--speed-percent", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.1 <= abs(args.delta_mm) <= 2.0:
        raise ValueError("--delta-mm magnitude must be between 0.1 and 2.0 mm")
    if not 1 <= args.speed_percent <= 3:
        raise ValueError("--speed-percent must be between 1 and 3")

    config = load_config(args.config)
    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(
        can_name=config.robot.can_interface,
        judge_flag=config.robot.official_can_adapter,
        can_auto_init=True,
        dh_is_offset=config.robot.dh_is_offset,
        start_sdk_joint_limit=True,
        start_sdk_gripper_limit=True,
    )
    piper.ConnectPort(piper_init=False, start_thread=True)
    cleanup_required = False
    try:
        time.sleep(0.5)
        start = read_pose_units(piper)
        target = start.copy()
        axis_index = {"x": 0, "y": 1, "z": 2}[args.axis]
        target[axis_index] += int(round(args.delta_mm * 1000.0))

        print("=== PIPER CARTESIAN DIAGNOSTIC ===")
        print("Read-only until explicit confirmation.")
        print_status(piper, "initial controller feedback")
        print("\nCurrent SDK pose [mm, deg]:", (start / 1000.0).tolist())
        print("Target  SDK pose [mm, deg]:", (target / 1000.0).tolist())
        print(f"Requested jog: SDK {args.axis.upper()} {args.delta_mm:+.3f} mm")
        print("Speed:", args.speed_percent, "%")
        print("Gripper: unchanged")
        print("Final behavior: hold the measured final joints with motors enabled")

        if not args.execute:
            print("\nDRY RUN ONLY: no enable or motion command was sent.")
            return
        if not args.confirm_motion:
            raise RuntimeError("Motion requires both --execute and --confirm-motion")

        print("\nClear the Cartesian jog direction and keep the emergency stop in hand.")
        print("Run only one Piper process.")
        typed = input(f"Type exactly {CONFIRM_TEXT} to continue: ").strip()
        if typed != CONFIRM_TEXT:
            raise RuntimeError("Confirmation text did not match; motion cancelled")

        cleanup_required = True
        enable_with_timeout(piper, timeout_s=5.0)
        print_status(piper, "after EnablePiper handshake")

        # EndPoseCtrl is a three-frame target register. Preload the measured
        # pose while still in MOVE J so switching to MOVE P cannot activate a
        # stale Cartesian target left by an earlier process.
        piper.EndPoseCtrl(*start.astype(int).tolist())
        time.sleep(0.1)

        # Intentionally do not issue MotionCtrl_1(track_ctrl=CONTINUE). This
        # isolates the normal MOVE P path from the suspected trajectory state.
        piper.MotionCtrl_2(0x01, 0x00, args.speed_percent, 0x00)
        time.sleep(0.2)
        print_mode_command(piper)
        print_status(piper, "after MOVE P request")

        status_after_mode_switch = arm_status_code(piper)
        if status_after_mode_switch != 0x00:
            name = STATUS_NAMES.get(status_after_mode_switch, "unknown")
            raise RuntimeError(
                "MOVE P mode switch failed after preloading the measured pose: "
                f"0x{status_after_mode_switch:02X} ({name}); jog target not sent"
            )

        piper.EndPoseCtrl(*target.astype(int).tolist())
        final, maximum_mm = wait_for_feedback(piper, start, args.timeout)
        delta = (final[:3] - start[:3]) / 1000.0
        target_error = float(np.linalg.norm(final[:3] - target[:3]) / 1000.0)
        print_status(piper, "after EndPoseCtrl")
        print("\n=== RESULT ===")
        print("Final SDK pose [mm, deg]:", (final / 1000.0).tolist())
        print("Feedback XYZ delta [mm]:", np.round(delta, 4).tolist())
        print("Maximum observed translation [mm]:", round(maximum_mm, 4))
        print("Position error to target [mm]:", round(target_error, 4))
        if maximum_mm < 0.20:
            print("FAIL: command was sent, but Cartesian feedback did not move 0.20 mm.")
        elif arm_status_code(piper) != 0x00:
            print("FAIL: movement/status changed, but the controller reports an error.")
        else:
            print("PASS: MOVE P / EndPoseCtrl produced measurable Cartesian motion.")
    except KeyboardInterrupt:
        print("\nInterrupted; holding the measured current joint position.")
    finally:
        if cleanup_required:
            held = hold_current_position(piper)
            print("Holding joints with motors enabled:", np.round(held, 3).tolist())
            print("WARNING: support the arm before any later DisableArm/power-off.")
        piper.DisconnectPort()


if __name__ == "__main__":
    main()
