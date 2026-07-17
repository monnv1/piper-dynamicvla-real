"""Read and print Piper gripper feedback without sending control commands."""

from __future__ import annotations

import argparse
import time
from datetime import datetime


FAULT_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "sensor_status",
    "driver_error_status",
)


def format_gripper_state(message, control_message=None) -> str:
    state = message.gripper_state
    foc = state.foc_status
    angle_units = int(state.grippers_angle)
    effort_units = int(state.grippers_effort)
    status_code = int(getattr(state, "status_code", 0))
    active_faults = [name for name in FAULT_FIELDS if bool(getattr(foc, name, False))]
    faults = ",".join(active_faults) if active_faults else "none"

    control_text = "ctrl_target=n/a ctrl_hz=0.0"
    if control_message is not None:
        control_hz = float(getattr(control_message, "Hz", 0.0))
        if control_hz > 0:
            control = control_message.gripper_ctrl
            control_text = (
                f"ctrl_target={int(control.grippers_angle) / 1000.0:.3f}mm "
                f"ctrl_code=0x{int(control.status_code):02X} "
                f"ctrl_set_zero=0x{int(control.set_zero):02X} "
                f"ctrl_hz={control_hz:.1f}"
            )

    return (
        f"{datetime.now().isoformat(timespec='milliseconds')} "
        f"position={angle_units / 1000.0:.3f}mm raw_position={angle_units} "
        f"effort={effort_units / 1000.0:.3f}Nm raw_effort={effort_units} "
        f"status=0x{status_code:02X} "
        f"enabled={bool(getattr(foc, 'driver_enable_status', False))} "
        f"homed={bool(getattr(foc, 'homing_status', False))} "
        f"faults={faults} "
        f"feedback_hz={float(getattr(message, 'Hz', 0.0)):.1f} "
        f"{control_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Piper gripper feedback")
    parser.add_argument("--can", default="can0", help="CAN interface (default: can0)")
    parser.add_argument(
        "--hz", type=float, default=1.0, help="Print frequency (default: 1 Hz)"
    )
    parser.add_argument(
        "--once", action="store_true", help="Print one valid feedback sample and exit"
    )
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("--hz must be positive")

    from piper_sdk import C_PiperInterface_V2

    piper = C_PiperInterface_V2(
        can_name=args.can,
        start_sdk_joint_limit=True,
        start_sdk_gripper_limit=True,
    )
    piper.ConnectPort(piper_init=False, start_thread=True)
    period = 1.0 / args.hz
    next_tick = time.monotonic()

    print(
        f"Reading Piper gripper feedback on {args.can} at {args.hz:g} Hz (Ctrl+C to stop)"
    )
    try:
        feedback_deadline = time.monotonic() + 3.0
        while True:
            message = piper.GetArmGripperMsgs()
            if float(getattr(message, "Hz", 0.0)) > 0:
                break
            if time.monotonic() >= feedback_deadline:
                raise TimeoutError("No Piper gripper feedback received within 3 seconds")
            time.sleep(0.02)

        while True:
            message = piper.GetArmGripperMsgs()
            control_message = piper.GetArmGripperCtrl()
            print(format_gripper_state(message, control_message), flush=True)
            if args.once:
                break
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        piper.DisconnectPort()


if __name__ == "__main__":
    main()
