from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from deploy.common.messages import ActionChunk, ScheduledAction


@dataclass
class SchedulerStats:
    accepted_chunks: int = 0
    bootstrap_chunks: int = 0
    reanchored_chunks: int = 0
    stale_chunks: int = 0
    expired_actions: int = 0
    executed_actions: int = 0


@dataclass(frozen=True)
class DispatchTiming:
    target_ns: int
    actual_ns: int
    lateness_ns: int
    skipped_intervals: int


class FixedRateGate:
    """Monotonic fixed-rate gate that never emits catch-up bursts."""

    def __init__(self, frequency_hz: float) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        self.period_ns = int(round(1_000_000_000 / frequency_hz))
        self._next_ns: int | None = None

    def arm(self, now_ns: int) -> None:
        self._next_ns = now_ns

    def ready(self, now_ns: int) -> bool:
        return self._next_ns is not None and now_ns >= self._next_ns

    def consume(self, now_ns: int) -> DispatchTiming:
        if not self.ready(now_ns):
            raise RuntimeError("Fixed-rate gate was consumed before its deadline")
        assert self._next_ns is not None
        target_ns = self._next_ns
        lateness_ns = max(0, now_ns - target_ns)
        skipped_intervals = lateness_ns // self.period_ns
        # A late control loop schedules the next action one full period from
        # now. It never immediately replays missed setpoints in a burst.
        if lateness_ns:
            self._next_ns = now_ns + self.period_ns
        else:
            self._next_ns = target_ns + self.period_ns
        return DispatchTiming(
            target_ns=target_ns,
            actual_ns=now_ns,
            lateness_ns=lateness_ns,
            skipped_intervals=int(skipped_intervals),
        )


class ActionScheduler:
    """Timestamp-aware replacement for the upstream LAAS queue."""

    def __init__(self, chunk_size: int, max_action_age_ms: float) -> None:
        self.chunk_size = chunk_size
        self.max_action_age_ns = int(max_action_age_ms * 1_000_000)
        self._actions: dict[int, ScheduledAction] = {}
        self._episode_id = ""
        self._last_source_index = -1
        self._last_executed_index = -1
        self._lock = threading.Lock()
        self.stats = SchedulerStats()

    def reset(self, episode_id: str) -> None:
        with self._lock:
            self._episode_id = episode_id
            self._actions.clear()
            self._last_source_index = -1
            self._last_executed_index = -1
            self.stats = SchedulerStats()

    def submit(
        self,
        chunk: ActionChunk,
        current_index: int,
        now_ns: int | None = None,
        reanchor: bool = False,
        max_actions: int | None = None,
    ) -> bool:
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        with self._lock:
            if chunk.episode_id != self._episode_id:
                self.stats.stale_chunks += 1
                return False
            if chunk.observation_index <= self._last_source_index:
                self.stats.stale_chunks += 1
                return False
            if now_ns - chunk.observation_timestamp_ns > self.max_action_age_ns:
                self.stats.stale_chunks += 1
                return False

            is_bootstrap = self.stats.accepted_chunks == 0
            if is_bootstrap or reanchor:
                # No older action chunk was running while the first inference
                # was in flight. The robot stayed at the observed pose, so
                # elapsed wall-clock steps must not expire the beginning of
                # this trajectory. Re-anchor action[0] to the current tick.
                skip = 0
                start_index = current_index
                self.stats.bootstrap_chunks += int(is_bootstrap)
                self.stats.reanchored_chunks += int(reanchor)
            else:
                # Once streaming is established, an older chunk fills the
                # inference gap and normal LAAS expiration is valid.
                skip = max(0, current_index - chunk.observation_index)
                self.stats.expired_actions += min(skip, len(chunk.actions))
                if skip >= len(chunk.actions):
                    self.stats.stale_chunks += 1
                    self._last_source_index = chunk.observation_index
                    return False
                start_index = chunk.observation_index + skip
            for target_index in list(self._actions):
                if target_index >= start_index:
                    del self._actions[target_index]

            selected_actions = chunk.actions[skip:]
            if max_actions is not None:
                selected_actions = selected_actions[:max_actions]
            for offset, action in enumerate(selected_actions):
                target_index = start_index + offset
                if target_index <= self._last_executed_index:
                    continue
                self._actions[target_index] = ScheduledAction(
                    episode_id=chunk.episode_id,
                    target_index=target_index,
                    source_observation_index=chunk.observation_index,
                    action=action.copy(),
                    source_state=(
                        None
                        if chunk.source_state is None
                        else chunk.source_state.copy()
                    ),
                )
            self._last_source_index = chunk.observation_index
            self.stats.accepted_chunks += 1
            return True

    def has_pending_actions(self) -> bool:
        with self._lock:
            return bool(self._actions)

    def pop_next(self) -> ScheduledAction | None:
        with self._lock:
            if not self._actions:
                return None
            target_index = min(self._actions)
            action = self._actions.pop(target_index)
            self._last_executed_index = max(self._last_executed_index, target_index)
            self.stats.executed_actions += 1
            return action

    def pop(self, current_index: int) -> ScheduledAction | None:
        with self._lock:
            for target_index in [key for key in self._actions if key < current_index]:
                del self._actions[target_index]
                self.stats.expired_actions += 1
            action = self._actions.pop(current_index, None)
            if action is not None:
                self._last_executed_index = current_index
                self.stats.executed_actions += 1
            return action
