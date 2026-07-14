"""Read and print Piper gripper feedback without sending control commands."""

from __future__ import annotations

import argparse
import time
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Piper gripper feedback")
    parser.add_argument("--can", default="can0", help="CAN interface (default: can0)")
    parser.add_argument(
        "--hz", type=float, default=1.0, help="Print frequency (default: 1 Hz)"
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
        while True:
            message = piper.GetArmGripperMsgs()
            state = message.gripper_state
            angle_units = int(state.grippers_angle)
            effort_units = int(state.grippers_effort)
            print(
                f"{datetime.now().isoformat(timespec='milliseconds')} "
                f"angle={angle_units / 1000.0:.3f} mm "
                f"({angle_units / 1_000_000.0:.6f} m, raw={angle_units}) "
                f"effort={effort_units / 1000.0:.3f} Nm "
                f"(raw={effort_units}) "
                f"foc_status={state.foc_status!r} "
                f"feedback_hz={float(getattr(message, 'Hz', 0.0)):.1f}"
            )
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
