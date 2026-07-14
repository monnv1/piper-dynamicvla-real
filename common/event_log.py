from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    return value


class EventLog:
    def __init__(self, output_dir: str | Path, episode_id: str) -> None:
        self.directory = Path(output_dir) / episode_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "events.jsonl"
        self._stream = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, event: str, **values) -> None:
        record = {
            "event": event,
            "wall_time": time.time(),
            "monotonic_ns": time.monotonic_ns(),
            **values,
        }
        with self._lock:
            self._stream.write(
                json.dumps(record, default=_json_value, ensure_ascii=False) + "\n"
            )
            self._stream.flush()

    def close(self) -> None:
        self._stream.close()

