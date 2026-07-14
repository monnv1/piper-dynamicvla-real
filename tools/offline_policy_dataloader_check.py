from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import utils.datasets
import utils.helpers
from policies.dynamicvla.modeling_dynamicvla import load_dynamicvla
from utils.instruction_generator import InstructionGenerator
from deploy.tools.offline_policy_rollout import jsonable, load_policy, wrap_euler_action


def dataset_root_from_episode(episode_path: Path) -> Path:
    return episode_path.parent.parent.parent


def episode_index_from_path(episode_path: Path) -> int:
    stem = episode_path.stem
    if not stem.startswith("episode_"):
        raise ValueError(f"Cannot infer episode index from {episode_path.name}")
    return int(stem.split("_")[-1])


def load_policy_and_config(checkpoint: Path, device: torch.device, weights: Path | None = None):
    config = json.loads((checkpoint / "config.json").read_text())
    policy = load_policy(checkpoint, device)
    if weights is not None:
        load_dynamicvla(policy, weights, device=str(device))
        policy.eval()
    return policy, config


def to_batch(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in item.items():
        if key.endswith("_is_pad"):
            continue
        if key == "task":
            batch[key] = [value]
        elif isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run policy on a sample produced by the training LeRobotDataset path."
    )
    parser.add_argument("--checkpoint", default="/data/checkpoints/piper_real_20x")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--weights", default=None, help="Optional safetensors file inside/outside checkpoint dir to evaluate instead of model.safetensors")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compare-steps", type=int, default=10)
    parser.add_argument("--task", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    episode_path = Path(args.episode).expanduser().resolve()
    dataset_root = dataset_root_from_episode(episode_path)
    episode_index = episode_index_from_path(episode_path)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    weights = Path(args.weights).expanduser().resolve() if args.weights else None
    policy, checkpoint_config = load_policy_and_config(checkpoint, device, weights)
    required_features = list(policy.config.input_features.keys()) + ["action"]
    delta_timestamps = {
        "observation": checkpoint_config.get("delta_timestamps", {}).get("observation", [-2, 0]),
        "action": list(range(int(policy.config.chunk_size))),
    }
    image_size = tuple(policy.config.input_features["observation.images.opst_cam"].shape[-2:])
    dataset = utils.datasets.LeRobotDataset(
        repo_id=dataset_root.name,
        root=dataset_root,
        split="train",
        delta_action=True,
        required_features=required_features,
        episodes=[episode_index],
        image_transforms=utils.datasets.ImageTransforms(image_size),
        delta_timestamps=delta_timestamps,
    )
    original_instruction_generator = InstructionGenerator.generate_instruction
    InstructionGenerator.generate_instruction = staticmethod(
        lambda metadata: metadata if isinstance(metadata, str) else str(metadata)
    )
    try:
        item = dataset[args.row]
    finally:
        InstructionGenerator.generate_instruction = original_instruction_generator
    if args.task is not None:
        item["task"] = args.task
    batch = to_batch(item, device)

    policy.reset()
    with torch.inference_mode():
        pred_delta = policy.predict_action_chunk(batch)[0].detach().cpu().float().numpy()

    gt_delta = item["action"].detach().cpu().float().numpy()
    source_state = item["observation.state"][-1].detach().cpu().float().numpy()
    action_dims = pred_delta.shape[-1] - 1
    pred_abs = pred_delta.copy()
    if policy.config.use_delta_action:
        pred_abs[:, :action_dims] += source_state[:action_dims]
        pred_abs = wrap_euler_action(pred_abs)
    gt_abs = gt_delta.copy()
    gt_abs[:, :action_dims] += source_state[:action_dims]
    gt_abs = wrap_euler_action(gt_abs)

    horizon = min(args.compare_steps, len(pred_delta), len(gt_delta))
    err_delta = pred_delta[:horizon, : gt_delta.shape[-1]] - gt_delta[:horizon]
    result = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "weights": None if weights is None else str(weights),
        "episode_index": episode_index,
        "row": args.row,
        "task": item["task"],
        "delta_timestamps": delta_timestamps,
        "source_state": source_state,
        "pred_delta_first": pred_delta[0],
        "gt_delta_first": gt_delta[0],
        "summary": {
            "pred_delta_xyz_mean_m": np.mean(pred_delta[:horizon, :3], axis=0),
            "gt_delta_xyz_mean_m": np.mean(gt_delta[:horizon, :3], axis=0),
            "delta_error_xyz_mean_m": np.mean(err_delta[:, :3], axis=0),
            "delta_error_xyz_norm_mean_m": float(np.mean(np.linalg.norm(err_delta[:, :3], axis=1))),
        },
        "comparison": [],
    }
    for i in range(horizon):
        result["comparison"].append(
            {
                "step": i,
                "pred_delta": pred_delta[i],
                "gt_delta": gt_delta[i],
                "pred_abs": pred_abs[i],
                "gt_abs": gt_abs[i],
                "delta_error": err_delta[i],
            }
        )

    print("=== OFFLINE POLICY DATALOADER CHECK ===")
    print("checkpoint:", checkpoint)
    print("dataset_root:", dataset_root)
    print("weights:", weights if weights is not None else checkpoint / "model.safetensors")
    print("episode_index:", episode_index, "row:", args.row)
    print("task:", item["task"])
    print("delta_timestamps:", delta_timestamps)
    print("source_state:", np.round(source_state, 5).tolist())
    print("pred raw delta first:", np.round(pred_delta[0], 5).tolist())
    print("gt dataloader delta first:", np.round(gt_delta[0], 5).tolist())
    print("\nstep | pred dXYZ mm | gt dXYZ mm | err dXYZ mm | pred abs XYZ | gt abs XYZ")
    for comp in result["comparison"]:
        print(
            f"{comp['step']:>4d}",
            np.round(comp["pred_delta"][:3] * 1000.0, 1).tolist(),
            np.round(comp["gt_delta"][:3] * 1000.0, 1).tolist(),
            np.round(comp["delta_error"][:3] * 1000.0, 1).tolist(),
            np.round(comp["pred_abs"][:3], 4).tolist(),
            np.round(comp["gt_abs"][:3], 4).tolist(),
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
