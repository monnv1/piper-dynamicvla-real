from __future__ import annotations

import argparse
import time

import numpy as np

from deploy.config import load_config
from deploy.devices.piper_frames import PiperFrameTransform
from deploy.targets import TRAINING_START_DEG, TRAINING_START_RAD


JOINT_LIMITS_DEG = np.asarray(
    [
        [-150.0, 150.0],
        [0.0, 180.0],
        [-170.0, 0.0],
        [-100.0, 100.0],
        [-70.0, 70.0],
        [-120.0, 120.0],
    ]
)

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

CONFIRM_TEXT = "MOVE_PIPER_SLOWLY_TO_TRAINING_START"


def arm_status_code(piper) -> int:
    wrapper = piper.GetArmStatus()
    status = getattr(wrapper, "arm_status", wrapper)
    return int(getattr(status, "arm_status", -1))


def read_joint_degrees(piper) -> tuple[np.ndarray, float]:
    message = piper.GetArmJointMsgs()
    joint = message.joint_state
    values = np.asarray(
        [
            joint.joint_1,
            joint.joint_2,
            joint.joint_3,
            joint.joint_4,
            joint.joint_5,
            joint.joint_6,
        ],
        dtype=np.float64,
    ) / 1000.0
    return values, float(message.Hz)


def require_normal_status(piper, context: str) -> None:
    code = arm_status_code(piper)
    if code != 0x00:
        name = STATUS_NAMES.get(code, "unknown")
        raise RuntimeError(f"Piper status is 0x{code:02X} ({name}) during {context}")


def validate_joint_values(values: np.ndarray, label: str, margin_deg: float = 0.0) -> None:
    if values.shape != (6,) or not np.isfinite(values).all():
        raise RuntimeError(f"{label} must contain six finite joint angles")
    lower = JOINT_LIMITS_DEG[:, 0] + margin_deg
    upper = JOINT_LIMITS_DEG[:, 1] - margin_deg
    invalid = np.flatnonzero((values < lower) | (values > upper))
    if invalid.size:
        details = ", ".join(
            f"J{i + 1}={values[i]:.3f} not in [{lower[i]:.1f}, {upper[i]:.1f}]"
            for i in invalid
        )
        raise RuntimeError(f"{label} violates joint limits: {details}")


def enable_with_timeout(piper, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        require_normal_status(piper, "enable")
        if piper.EnablePiper():
            return
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for all Piper motors to enable")


def hold_current_position(piper) -> np.ndarray:
    """Replace the active target with the measured joint position.

    Piper DisableArm/DisablePiper removes motor support and can let the arm
    fall under gravity. Cleanup therefore leaves the motors enabled in MOVE J
    position mode. Disconnecting the SDK does not intentionally disable them.
    """
    current_deg, _ = read_joint_degrees(piper)
    validate_joint_values(current_deg, "hold joints")
    target_units = np.rint(current_deg * 1000.0).astype(int)
    piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)
    piper.JointCtrl(*target_units.tolist())
    time.sleep(0.2)
    return current_deg


def resume_position_control(piper) -> np.ndarray:
    """Release a previous track-pause/terminate state at the current pose."""
    current_deg, _ = read_joint_degrees(piper)
    # Queue the measured pose before issuing CONTINUE, so releasing an old
    # termination state cannot resume a stale distant target.
    send_joint_target(piper, current_deg, speed_percent=1)
    piper.MotionCtrl_1(0x00, 0x02, 0x00)  # continue trajectory execution
    send_joint_target(piper, current_deg, speed_percent=1)
    time.sleep(0.2)
    require_normal_status(piper, "resume position control")
    return current_deg


def send_joint_target(piper, target_deg: np.ndarray, speed_percent: int) -> None:
    target_units = np.rint(target_deg * 1000.0).astype(int)
    # CAN control + MOVE J + position/velocity mode.
    piper.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
    piper.JointCtrl(*target_units.tolist())


def sample_model_tcp_path(
    current_deg: np.ndarray,
    dh_is_offset: int,
    frames: PiperFrameTransform,
    samples: int = 51,
) -> np.ndarray:
    from piper_sdk import C_PiperForwardKinematics

    fk = C_PiperForwardKinematics(dh_is_offset)
    current_rad = np.radians(current_deg)
    positions = []
    for fraction in np.linspace(0.0, 1.0, samples):
        joints = current_rad + (TRAINING_START_RAD - current_rad) * fraction
        sdk_pose = np.asarray(fk.CalFK(joints)[-1], dtype=np.float64)
        model_position, _ = frames.sdk_to_model_pose(
            sdk_pose[:3] / 1000.0,
            np.radians(sdk_pose[3:]),
        )
        positions.append(model_position)
    return np.stack(positions)


def validate_tcp_path(
    positions: np.ndarray, workspace_min: np.ndarray, workspace_max: np.ndarray
) -> None:
    invalid = np.flatnonzero(
        np.any((positions < workspace_min) | (positions > workspace_max), axis=1)
    )
    if invalid.size:
        index = int(invalid[0])
        raise RuntimeError(
            "Approximate MOVE J FK path exits the configured model-TCP workspace "
            f"at sample {index}: {positions[index].tolist()}"
        )


def print_plan(
    current_deg: np.ndarray, speed_percent: int, tcp_path: np.ndarray
) -> None:
    delta = TRAINING_START_DEG - current_deg
    print("\n=== READ THIS PLAN BEFORE MOVING ===")
    print("Current joints (deg):", np.round(current_deg, 3).tolist())
    print("Target joints  (deg):", np.round(TRAINING_START_DEG, 3).tolist())
    print("Joint deltas   (deg):", np.round(delta, 3).tolist())
    print("Largest joint delta:", round(float(np.max(np.abs(delta))), 3), "deg")
    print("MOVE J speed:", speed_percent, "%")
    print("Gripper: close to 0 mm at return start and keep closed")
    print("Approx model-TCP path min (m):", np.round(tcp_path.min(axis=0), 4).tolist())
    print("Approx model-TCP path max (m):", np.round(tcp_path.max(axis=0), 4).tolist())
    print("SDK-FK model TCP at target (m):", np.round(tcp_path[-1], 4).tolist())
    print("State-machine Cartesian INIT target (m): [0.373, 0.0, 0.271]")
    print(
        "WARNING: joint-space interpolation does not check the table, cameras, "
        "cables, self-collision, or swept-volume collision."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slowly move Piper to the DynamicVLA simulation start joints"
    )
    parser.add_argument(
        "--config", default="deploy/configs/piper_gemini_d435i.yaml"
    )
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=5,
        help="MOVE J speed percentage; hard limited to 1..10 (default: 5)",
    )
    parser.add_argument(
        "--can-interface",
        default=None,
        help="Override robot.can_interface from config, e.g. can1 for the leader arm",
    )
    parser.add_argument(
        "--teaching-after",
        action="store_true",
        help="After reaching the target, request Piper drag-teaching mode",
    )
    parser.add_argument(
        "--free-after",
        action="store_true",
        help="After reaching the target, disable motors so the arm is unlocked. WARNING: the arm may lose support.",
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--tolerance-deg", type=float, default=1.5)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually enable and move; omission is a read-only dry run",
    )
    parser.add_argument(
        "--confirm-motion",
        action="store_true",
        help="Second motion gate; an interactive confirmation is still required",
    )
    args = parser.parse_args()

    if not 1 <= args.speed_percent <= 10:
        raise ValueError("--speed-percent must be in the hard-limited range 1..10")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if not 0.2 <= args.tolerance_deg <= 3.0:
        raise ValueError("--tolerance-deg must be between 0.2 and 3.0")
    if args.teaching_after and args.free_after:
        raise ValueError("Choose only one of --teaching-after or --free-after")

    config = load_config(args.config)
    validate_joint_values(TRAINING_START_DEG, "training target", margin_deg=0.5)
    frames = PiperFrameTransform(
        config.robot.sdk_to_model_translation_m,
        config.robot.sdk_to_model_euler_xyz_rad,
    )

    from piper_sdk import C_PiperInterface_V2

    can_interface = args.can_interface or config.robot.can_interface
    print("Using CAN interface:", can_interface)

    piper = C_PiperInterface_V2(
        can_name=can_interface,
        judge_flag=config.robot.official_can_adapter,
        can_auto_init=True,
        dh_is_offset=config.robot.dh_is_offset,
        start_sdk_joint_limit=True,
        start_sdk_gripper_limit=True,
    )

    motion_started = False
    piper.ConnectPort(piper_init=False, start_thread=True)
    try:
        time.sleep(1.0)
        current_deg, feedback_hz = read_joint_degrees(piper)
        if feedback_hz <= 0:
            raise RuntimeError("No Piper joint feedback")
        require_normal_status(piper, "preflight")
        validate_joint_values(current_deg, "current joints")
        tcp_path = sample_model_tcp_path(current_deg, config.robot.dh_is_offset, frames)
        validate_tcp_path(
            tcp_path,
            np.asarray(config.safety.workspace_min_m),
            np.asarray(config.safety.workspace_max_m),
        )
        print("Joint feedback Hz:", feedback_hz)
        print_plan(current_deg, args.speed_percent, tcp_path)

        if not args.execute:
            print("\nDRY RUN ONLY: no motor was enabled and no command was sent.")
            print("Re-run with --execute --confirm-motion only after physical inspection.")
            return
        if not args.confirm_motion:
            raise RuntimeError("Motion requires both --execute and --confirm-motion")

        print("\nBefore confirming:")
        print("- Clear all people, objects and loose cables from the swept volume.")
        print("- Verify the arm is mounted upright and the table will not be struck.")
        print("- Keep one operator's hand on the physical emergency stop.")
        print("- Do not run deploy.run or another Piper process concurrently.")
        typed = input(f"Type exactly {CONFIRM_TEXT} to move: ").strip()
        if typed != CONFIRM_TEXT:
            raise RuntimeError("Confirmation text did not match; motion cancelled")

        # Mark cleanup as required before the first enable command. This also
        # guarantees a hold-position command if enable feedback itself times out.
        motion_started = True
        enable_with_timeout(piper, timeout_s=5.0)
        piper.GripperCtrl(0, 1000, 0x01, 0x00)
        time.sleep(0.2)
        resumed = resume_position_control(piper)
        print("Position control active at:", np.round(resumed, 3).tolist())
        def move_phase(target_deg: np.ndarray, label: str) -> None:
            print(f"\nMoving phase: {label}")
            deadline = time.monotonic() + args.timeout
            settled_samples = 0
            last_print = 0.0
            best_error = float("inf")
            best_error_time = time.monotonic()

            while time.monotonic() < deadline:
                require_normal_status(piper, "motion")
                send_joint_target(piper, target_deg, args.speed_percent)
                current_deg, feedback_hz = read_joint_degrees(piper)
                validate_joint_values(current_deg, "feedback joints")
                error = target_deg - current_deg
                max_error = float(np.max(np.abs(error)))

                now = time.monotonic()
                if now - last_print >= 0.5:
                    print(
                        "joints_deg=",
                        np.round(current_deg, 2).tolist(),
                        "target_deg=",
                        np.round(target_deg, 2).tolist(),
                        "max_error_deg=",
                        round(max_error, 3),
                        "status=normal",
                    )
                    last_print = now

                if max_error < best_error - 0.2:
                    best_error = max_error
                    best_error_time = now
                elif now - best_error_time > 6.0 and max_error > args.tolerance_deg:
                    raise RuntimeError(
                        "Joint feedback is not converging for 6s. Stop sending commands; "
                        "check CAN send errors, emergency stop, arm fault state, and whether another process owns this CAN bus."
                    )

                if max_error <= args.tolerance_deg:
                    settled_samples += 1
                    if settled_samples >= 10:
                        print(f"Phase reached: {label}")
                        return
                else:
                    settled_samples = 0
                time.sleep(0.1)

            raise TimeoutError(
                f"Motion phase '{label}' did not settle within {args.timeout:.1f}s; arm will hold position"
            )

        phase1 = TRAINING_START_DEG.copy()
        phase1[4] = resumed[4]
        move_phase(phase1, "J1/J2/J3/J4/J6 first, keep J5 current")
        move_phase(TRAINING_START_DEG, "J5 last")
        piper.GripperCtrl(0, 1000, 0x01, 0x00)
        print("Target reached and stable within tolerance.")
    except KeyboardInterrupt:
        print("\nInterrupted; terminating motion and holding the current position.")
    finally:
        if motion_started:
            if args.free_after:
                print("WARNING: disabling motors; support the arm before it loses motor holding torque.")
                piper.MotionCtrl_2(0x00, 0x01, 1, 0x00)
                time.sleep(0.1)
                piper.DisablePiper()
                time.sleep(0.3)
                print("Arm motors disabled; arm should be unlocked if brakes/hardware permit it.")
            elif args.teaching_after:
                # Request drag teaching mode. Some firmware/hardware states may
                # reject this; print feedback so the operator can verify it.
                piper.MotionCtrl_2(0x00, 0x01, 1, 0x00)
                time.sleep(0.1)
                piper.MotionCtrl_1(0x00, 0x00, 0x01)
                time.sleep(0.3)
                status = piper.GetArmStatus().arm_status
                print(
                    "Arm drag-teaching request sent:",
                    "ctrl_mode=", status.ctrl_mode,
                    "teach_status=", status.teach_status,
                )
            else:
                held = hold_current_position(piper)
                print("Holding joints with motors enabled:", np.round(held, 3).tolist())
                print("WARNING: support the arm before any later DisableArm/power-off.")
        piper.DisconnectPort()


if __name__ == "__main__":
    main()
