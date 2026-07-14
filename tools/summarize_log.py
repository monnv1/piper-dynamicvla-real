from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize deploy events.jsonl")
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    counts = collections.Counter()
    inference_times: list[float] = []
    skipped_steps: list[int] = []
    with args.log.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            counts[record["event"]] += 1
            if record["event"] == "action_chunk":
                inference_times.append(record["inference_seconds"])
                skipped_steps.append(record["index"] - record["source_index"])
    print("events:", dict(counts))
    if inference_times:
        print(
            "inference_seconds:",
            {
                "mean": float(np.mean(inference_times)),
                "p95": float(np.percentile(inference_times, 95)),
                "max": float(np.max(inference_times)),
            },
        )
        print(
            "laas_skipped_steps:",
            {
                "mean": float(np.mean(skipped_steps)),
                "max": int(np.max(skipped_steps)),
            },
        )


if __name__ == "__main__":
    main()

