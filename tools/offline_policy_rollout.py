from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import utils.helpers
from deploy.policy.inference_worker import fit_feature_dimension


CAMERA_KEYS = ["observation.images.opst_cam", "observation.images.wrist_cam"]


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


def dataset_root_from_episode(episode_path: Path) -> Path:
    # <root>/data/chunk-000/episode_000000.parquet
    return episode_path.parent.parent.parent


def episode_index_from_path(episode_path: Path) -> int:
    stem = episode_path.stem
    if not stem.startswith("episode_"):
        raise ValueError(f"Cannot infer episode index from {episode_path.name}")
    return int(stem.split("_")[-1])


def read_parquet_columns(episode_path: Path) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq  # type: ignore

    table = pq.read_table(
        episode_path,
        columns=["observation.state", "action", "task_index", "frame_index"],
    )
    return {
        "state": np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32),
        "action": np.asarray(table.column("action").to_pylist(), dtype=np.float32),
        "task_index": np.asarray(table.column("task_index").to_pylist(), dtype=np.int64).reshape(-1),
        "frame_index": np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64).reshape(-1),
    }


def load_tasks(dataset_root: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    path = dataset_root / "meta" / "tasks.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tasks[int(record["task_index"])] = record["task"]
    return tasks


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def load_image_history(
    dataset_root: Path,
    episode_index: int,
    row_indices: list[int],
    video_frame_indices: list[int],
) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for key in CAMERA_KEYS:
        video_path = (
            dataset_root
            / "videos"
            / "chunk-000"
            / key
            / f"episode_{episode_index:06d}.mp4"
        )
        frames = [read_video_frame(video_path, frame_index) for frame_index in video_frame_indices]
        images[key] = np.stack(frames, axis=0)
    return images


def apply_image_ablation(
    images: dict[str, np.ndarray],
    *,
    swap_cameras: bool,
    zero_images: bool,
    zero_camera: str | None,
) -> dict[str, np.ndarray]:
    edited = {key: value.copy() for key, value in images.items()}
    if swap_cameras:
        edited[CAMERA_KEYS[0]], edited[CAMERA_KEYS[1]] = (
            edited[CAMERA_KEYS[1]],
            edited[CAMERA_KEYS[0]],
        )
    if zero_images:
        for key in edited:
            edited[key].fill(0)
    if zero_camera:
        key = zero_camera
        if not key.startswith("observation.images."):
            key = f"observation.images.{key}"
        if key not in edited:
            raise KeyError(f"Unknown camera for --zero-camera: {zero_camera}")
        edited[key].fill(0)
    return edited


def load_policy(checkpoint: Path, device: torch.device):
    config_path = checkpoint / "config.json"
    checkpoint_config = json.loads(config_path.read_text())
    policy_config = utils.helpers.get_policy_cfg(cfg_file=config_path)
    policy_class = utils.helpers.get_policy_class(checkpoint_config["type"])
    policy = policy_class.from_pretrained(checkpoint, config=policy_config)
    policy.eval()
    return policy.to(device)


def build_batch(policy, images: dict[str, np.ndarray], states: np.ndarray, task: str, device: torch.device) -> tuple[dict[str, Any], np.ndarray]:
    batch: dict[str, Any] = {}
    for key, frames in images.items():
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        expected_height, expected_width = policy.config.input_features[key].shape[-2:]
        if tensor.shape[-2:] != (expected_height, expected_width):
            tensor = F.interpolate(
                tensor,
                size=(expected_height, expected_width),
                mode="bilinear",
                align_corners=False,
            )
        batch[key] = tensor.contiguous().unsqueeze(0).to(device)

    expected_state_dimension = policy.config.input_features["observation.state"].shape[0]
    fitted_states = fit_feature_dimension(states, expected_state_dimension)
    batch["observation.state"] = torch.from_numpy(fitted_states).float().unsqueeze(0).to(device)
    batch["task"] = [task]
    return batch, fitted_states


def wrap_euler_action(actions: np.ndarray) -> np.ndarray:
    wrapped = actions.copy()
    wrapped[..., 3] = wrapped[..., 3] % (2.0 * np.pi)
    wrapped[..., 5] = wrapped[..., 5] % (2.0 * np.pi)
    return wrapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a DynamicVLA checkpoint offline on recorded real episode frames/states."
    )
    parser.add_argument("--checkpoint", default="/data/checkpoints/piper_real_20x")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--history", nargs="+", type=int, default=[-2, 0])
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare-steps", type=int, default=10)
    parser.add_argument("--swap-cameras", action="store_true")
    parser.add_argument("--zero-images", action="store_true")
    parser.add_argument("--zero-camera", choices=["opst_cam", "wrist_cam", "observation.images.opst_cam", "observation.images.wrist_cam"], default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_path = Path(args.episode)
    dataset_root = dataset_root_from_episode(episode_path)
    episode_index = episode_index_from_path(episode_path)
    data = read_parquet_columns(episode_path)
    n_rows = len(data["action"])
    if not 0 <= args.row < n_rows:
        raise IndexError(f"--row {args.row} out of range [0, {n_rows})")
    if not args.history or args.history[-1] != 0:
        raise ValueError("--history must end in 0")

    selected_rows = [max(0, min(n_rows - 1, args.row + rel)) for rel in args.history]
    selected_frame_indices = [int(data["frame_index"][row]) for row in selected_rows]
    tasks = load_tasks(dataset_root)
    task = args.task or tasks[int(data["task_index"][args.row])]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    policy = load_policy(Path(args.checkpoint), device)
    images = load_image_history(dataset_root, episode_index, selected_rows, selected_frame_indices)
    images = apply_image_ablation(
        images,
        swap_cameras=args.swap_cameras,
        zero_images=args.zero_images,
        zero_camera=args.zero_camera,
    )
    states = data["state"][selected_rows]
    batch, fitted_states = build_batch(policy, images, states, task, device)
    source_state = fitted_states[-1]

    policy.reset()
    with torch.inference_mode():
        predicted_delta = policy.predict_action_chunk(batch)[0].detach().cpu().float().numpy()
    action_dims = predicted_delta.shape[-1] - 1
    predicted_abs = predicted_delta.copy()
    if policy.config.use_delta_action:
        predicted_abs[:, :action_dims] += source_state[:action_dims]
        predicted_abs = wrap_euler_action(predicted_abs)

    horizon = min(args.compare_steps, len(predicted_abs), n_rows - args.row)
    dataset_actions = data["action"][args.row : args.row + horizon]
    dataset_states = data["state"][args.row : args.row + horizon]
    dataset_delta_to_source = dataset_actions[:, :action_dims] - source_state[:action_dims]
    dataset_delta_to_own_state = dataset_actions[:, :action_dims] - dataset_states[:, :action_dims]
    pred_delta_to_source = predicted_abs[:horizon, :action_dims] - source_state[:action_dims]
    abs_error = predicted_abs[:horizon, : dataset_actions.shape[-1]] - dataset_actions

    result = {
        "checkpoint": str(args.checkpoint),
        "episode": str(episode_path),
        "episode_index": episode_index,
        "row": args.row,
        "history": args.history,
        "selected_rows": selected_rows,
        "selected_frame_indices": selected_frame_indices,
        "swap_cameras": args.swap_cameras,
        "zero_images": args.zero_images,
        "zero_camera": args.zero_camera,
        "task": task,
        "device": str(device),
        "source_state": source_state,
        "dataset_current_state": data["state"][args.row],
        "dataset_current_action": data["action"][args.row],
        "predicted_delta_first": predicted_delta[0],
        "predicted_abs_first": predicted_abs[0],
        "dataset_action_minus_source_first": dataset_delta_to_source[0],
        "dataset_action_minus_own_state_first": dataset_delta_to_own_state[0],
        "comparison": [],
        "summary": {
            "pred_delta_to_source_xyz_mean_m": np.mean(pred_delta_to_source[:, :3], axis=0),
            "dataset_delta_to_source_xyz_mean_m": np.mean(dataset_delta_to_source[:, :3], axis=0),
            "dataset_delta_to_own_state_xyz_mean_m": np.mean(dataset_delta_to_own_state[:, :3], axis=0),
            "abs_error_xyz_mean_m": np.mean(abs_error[:, :3], axis=0),
            "abs_error_xyz_norm_mean_m": float(np.mean(np.linalg.norm(abs_error[:, :3], axis=1))),
        },
    }
    for i in range(horizon):
        result["comparison"].append(
            {
                "step": i,
                "row": args.row + i,
                "predicted_abs": predicted_abs[i],
                "dataset_action": dataset_actions[i],
                "pred_delta_to_source": pred_delta_to_source[i],
                "dataset_delta_to_source": dataset_delta_to_source[i],
                "dataset_delta_to_own_state": dataset_delta_to_own_state[i],
                "abs_error": abs_error[i],
            }
        )

    print("=== OFFLINE POLICY ROLLOUT ===")
    print("checkpoint:", args.checkpoint)
    print("episode:", episode_path)
    print("row:", args.row, "history rows:", selected_rows, "video frames:", selected_frame_indices)
    print("ablations:", {"swap_cameras": args.swap_cameras, "zero_images": args.zero_images, "zero_camera": args.zero_camera})
    print("task:", task)
    print("source_state:", np.round(source_state, 5).tolist())
    print("dataset action row:", np.round(data["action"][args.row], 5).tolist())
    print("predicted raw delta first:", np.round(predicted_delta[0], 5).tolist())
    print("predicted abs first:", np.round(predicted_abs[0], 5).tolist())
    print("predicted delta-to-source first:", np.round(pred_delta_to_source[0], 5).tolist())
    print("dataset delta-to-source first:", np.round(dataset_delta_to_source[0], 5).tolist())
    print("\nstep | pred dXYZ mm | data dXYZ mm | pred abs XYZ | data abs XYZ | err XYZ mm")
    for item in result["comparison"]:
        print(
            f"{item['step']:>4d}",
            np.round(item["pred_delta_to_source"][:3] * 1000.0, 1).tolist(),
            np.round(item["dataset_delta_to_source"][:3] * 1000.0, 1).tolist(),
            np.round(item["predicted_abs"][:3], 4).tolist(),
            np.round(item["dataset_action"][:3], 4).tolist(),
            np.round(item["abs_error"][:3] * 1000.0, 1).tolist(),
        )
    print("\nsummary:")
    for key, value in result["summary"].items():
        if isinstance(value, np.ndarray):
            print(key, np.round(value, 6).tolist())
        else:
            print(key, value)

    if args.output:
        Path(args.output).write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False))
        print("wrote:", args.output)


if __name__ == "__main__":
    main()
