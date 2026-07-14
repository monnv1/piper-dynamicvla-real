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


CONFIRM_TEXT = "REPLAY_REAL_EPISODE"


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
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_event(handle, event: dict[str, object]) -> None:
    handle.write(json.dumps(jsonable(event), ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def load_parquet_actions(path: Path) -> np.ndarray:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as error:
        raise RuntimeError(
            "pyarrow is required to read LeRobot parquet episodes in this environment"
        ) from error

    table = pq.read_table(path, columns=["action"])
    values = table.column("action").to_pylist()
    actions = np.asarray(values, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError(f"Expected action shape [T,7], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError("Episode action contains NaN or Inf")
    return actions


def load_parquet_states(path: Path) -> np.ndarray | None:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None
    try:
        table = pq.read_table(path, columns=["observation.state"])
    except Exception:
        return None
    values = table.column("observation.state").to_pylist()
    states = np.asarray(values, dtype=np.float64)
    return states if states.ndim == 2 and states.shape[1] >= 6 else None


def wait_until_reached(
    robot: PiperRobot,
    timeout_s: float,
    poll_period_s: float,
) -> tuple[bool, int, int | None]:
    deadline = time.monotonic() + timeout_s
    motion_status: int | None = None
    while time.monotonic() < deadline:
        state = robot.latest_state()
        if state is not None:
            motion_status = state.motion_status
            if motion_status == 0x00:
                return True, motion_status, state.arm_status
        time.sleep(poll_period_s)
    return False, motion_status if motion_status is not None else -1, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay absolute model-TCP actions from a real LeRobot parquet episode."
    )
    parser.add_argument("--config", default="deploy/configs/piper_sequential.yaml")
    parser.add_argument("--episode", required=True, help="Path to episode_XXXXXX.parquet")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max steps to replay (0 = all)")
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument("--step-timeout", type=float, default=10.0)
    parser.add_argument("--poll-hz", type=float, default=50.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-motion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_path = Path(args.episode)
    if not episode_path.is_file():
        raise FileNotFoundError(episode_path)
    if args.start < 0 or args.stride <= 0 or args.max_steps < 0:
        raise ValueError("--start must be >=0, --stride must be positive, --max-steps must be >=0")
    if not 1 <= args.speed_percent <= 10:
        raise ValueError("--speed-percent must be in [1,10]")
    if args.poll_hz <= 0:
        raise ValueError("--poll-hz must be positive")

    actions = load_parquet_actions(episode_path)
    states = load_parquet_states(episode_path)
    end = len(actions) if args.end is None else min(args.end, len(actions))
    indices = list(range(args.start, end, args.stride))
    if args.max_steps > 0:
        indices = indices[: args.max_steps]
    if not indices:
        raise RuntimeError("No replay indices selected")

    config = load_config(args.config)
    config.robot.control_backend = "host_ik_move_j"
    config.robot.command_speed_percent = args.speed_percent
    config.robot.command_gripper = True

    output = Path(args.output) if args.output else Path(config.runtime.output_dir) / (
        "replay-real-" + time.strftime("%Y%m%d-%H%M%S") + ".jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    print("=== REAL EPISODE REPLAY ===")
    print("Episode:", episode_path)
    print("Output JSONL:", output)
    print("Selected indices:", indices[:10], "... total", len(indices))
    print("Speed percent:", config.robot.command_speed_percent)
    print("Step timeout:", args.step_timeout, "s")
    print("This sends dataset absolute model-TCP actions, one selected row at a time.")
    print("Motion completion is detected via Piper GetArmStatus().motion_status.")

    if not args.execute:
        print("\nDRY RUN: no robot connection or motion. First selected actions:")
        for index in indices[:10]:
            delta = None if states is None else actions[index, :6] - states[index, :6]
            print(
                index,
                "action=", np.round(actions[index], 5).tolist(),
                "action-state=", None if delta is None else np.round(delta, 5).tolist(),
            )
        return
    if not args.confirm_motion:
        raise RuntimeError("Motion requires --execute and --confirm-motion")
    typed = input(f"Type exactly {CONFIRM_TEXT} to replay: ").strip()
    if typed != CONFIRM_TEXT:
        raise RuntimeError("Confirmation text did not match; cancelled")

    robot = PiperRobot(config.robot)
    motion_started = False
    with output.open("w", encoding="utf-8") as log:
        try:
            robot.start()
            robot.enable_motion()
            motion_started = True
            write_event(
                log,
                {
                    "event": "replay_start",
                    "episode": str(episode_path),
                    "indices": indices,
                    "config": args.config,
                    "speed_percent": config.robot.command_speed_percent,
                },
            )
            poll_period = 1.0 / args.poll_hz
            for count, index in enumerate(indices):
                action = actions[index].copy()
                dataset_state = None if states is None else states[index].copy()
                try:
                    prepared = robot.preview_action(action)
                    command = robot.command_action(action, prepared=prepared)
                except HostIKError as error:
                    write_event(log, {
                        "event": "replay_step",
                        "count": count, "row_index": index,
                        "dataset_action": action, "dataset_state": dataset_state,
                        "reached": False, "host_ik_error": error.diagnostics,
                    })
                    print(f"[{count+1}/{len(indices)}] row={index} IK rejected, stopping")
                    break

                print(
                    f"[{count+1}/{len(indices)}] row={index} "
                    f"target_pos={np.round(action[:3], 4).tolist()} "
                    f"sent — waiting for motion_status==0"
                )
                reached, motion_status, arm_status = wait_until_reached(
                    robot, args.step_timeout, poll_period,
                )
                print(
                    f"  reached={reached} motion_status=0x{motion_status:02X}"
                    + (f" arm_status=0x{arm_status:02X}" if arm_status is not None else "")
                )
                write_event(log, {
                    "event": "replay_step",
                    "count": count, "row_index": index,
                    "dataset_action": action, "dataset_state": dataset_state,
                    "dataset_action_minus_state": None
                    if dataset_state is None
                    else action[:6] - dataset_state[:6],
                    "reached": reached,
                    "motion_status": motion_status,
                    "arm_status": arm_status,
                    "command_result": command,
                })
                if not reached:
                    print("  stopping replay because target was not reached within timeout")
                    break
            write_event(log, {"event": "replay_stop"})
        finally:
            if motion_started:
                print("Stopping: holding measured current joints and disconnecting.")
            robot.stop()


if __name__ == "__main__":
    main()
