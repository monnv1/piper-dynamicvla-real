import numpy as np

from deploy.common.messages import ActionChunk
from deploy.policy.action_scheduler import ActionScheduler, FixedRateGate


def chunk(origin: int, now_ns: int = 1_000_000_000, episode: str = "episode"):
    actions = np.arange(20 * 7, dtype=np.float32).reshape(20, 7) + origin * 1000
    return ActionChunk(
        episode_id=episode,
        observation_index=origin,
        observation_timestamp_ns=now_ns,
        completed_timestamp_ns=now_ns,
        actions=actions,
        inference_seconds=0.2,
    )


def test_first_chunk_is_reanchored_without_skipping():
    scheduler = ActionScheduler(20, max_action_age_ms=800)
    scheduler.reset("episode")
    assert scheduler.submit(chunk(0), current_index=5, now_ns=1_100_000_000)
    action = scheduler.pop(5)
    assert action is not None
    np.testing.assert_array_equal(action.action, chunk(0).actions[0])
    next_action = scheduler.pop(6)
    assert next_action is not None
    np.testing.assert_array_equal(next_action.action, chunk(0).actions[1])
    assert scheduler.stats.bootstrap_chunks == 1
    assert scheduler.stats.expired_actions == 0


def test_later_chunk_discards_expired_prefix():
    scheduler = ActionScheduler(20, max_action_age_ms=800)
    scheduler.reset("episode")
    scheduler.submit(chunk(0), current_index=5, now_ns=1_100_000_000)
    scheduler.pop(5)
    scheduler.pop(6)
    assert scheduler.submit(chunk(4), current_index=7, now_ns=1_100_000_000)
    action = scheduler.pop(7)
    assert action is not None
    np.testing.assert_array_equal(action.action, chunk(4).actions[3])
    assert scheduler.stats.expired_actions == 3


def test_new_chunk_replaces_future_actions():
    scheduler = ActionScheduler(20, max_action_age_ms=800)
    scheduler.reset("episode")
    scheduler.submit(chunk(0), current_index=5, now_ns=1_100_000_000)
    scheduler.pop(5)
    scheduler.submit(chunk(4), current_index=6, now_ns=1_100_000_000)
    action = scheduler.pop(6)
    np.testing.assert_array_equal(action.action, chunk(4).actions[2])


def test_rejects_wrong_episode_and_old_chunk():
    scheduler = ActionScheduler(20, max_action_age_ms=800)
    scheduler.reset("episode")
    assert not scheduler.submit(
        chunk(0, episode="old"), current_index=1, now_ns=1_100_000_000
    )
    assert scheduler.submit(chunk(2), current_index=2, now_ns=1_100_000_000)
    assert not scheduler.submit(chunk(1), current_index=3, now_ns=1_100_000_000)


def test_rejects_too_old_chunk():
    scheduler = ActionScheduler(20, max_action_age_ms=100)
    scheduler.reset("episode")
    assert not scheduler.submit(chunk(0), current_index=2, now_ns=1_200_000_000)


def test_sequential_chunk_reanchors_and_limits_trusted_prefix():
    scheduler = ActionScheduler(20, max_action_age_ms=800)
    scheduler.reset("episode")
    scheduler.submit(chunk(0), current_index=5, now_ns=1_100_000_000)
    for index in range(5, 25):
        scheduler.pop(index)

    assert scheduler.submit(
        chunk(30),
        current_index=40,
        now_ns=1_100_000_000,
        reanchor=True,
        max_actions=5,
    )
    for offset in range(5):
        action = scheduler.pop(40 + offset)
        assert action is not None
        np.testing.assert_array_equal(action.action, chunk(30).actions[offset])
    assert scheduler.pop(45) is None
    assert scheduler.stats.reanchored_chunks == 1
    assert scheduler.stats.expired_actions == 0


def test_pop_next_preserves_actions_while_wall_clock_advances():
    scheduler = ActionScheduler(20, max_action_age_ms=800)
    scheduler.reset("episode")
    assert scheduler.submit(
        chunk(0),
        current_index=10,
        now_ns=1_100_000_000,
        reanchor=True,
        max_actions=3,
    )

    first = scheduler.pop_next()
    assert first is not None
    np.testing.assert_array_equal(first.action, chunk(0).actions[0])
    # No pop occurs while the robot is moving. The next action must remain even
    # if many control-loop ticks pass before feedback reports completion.
    assert scheduler.has_pending_actions()
    second = scheduler.pop_next()
    assert second is not None
    np.testing.assert_array_equal(second.action, chunk(0).actions[1])
    third = scheduler.pop_next()
    assert third is not None
    np.testing.assert_array_equal(third.action, chunk(0).actions[2])
    assert scheduler.pop_next() is None
    assert scheduler.stats.expired_actions == 0


def test_fixed_rate_gate_dispatches_at_requested_period():
    gate = FixedRateGate(40.0)
    gate.arm(1_000_000_000)

    assert gate.ready(1_000_000_000)
    first = gate.consume(1_000_000_000)
    assert first.lateness_ns == 0
    assert not gate.ready(1_024_999_999)
    assert gate.ready(1_025_000_000)


def test_fixed_rate_gate_does_not_catch_up_after_overrun():
    gate = FixedRateGate(40.0)
    gate.arm(1_000_000_000)
    gate.consume(1_000_000_000)

    late = gate.consume(1_100_000_000)

    assert late.skipped_intervals == 3
    assert not gate.ready(1_124_999_999)
    assert gate.ready(1_125_000_000)


def test_fixed_rate_gate_preserves_spacing_after_small_delay():
    gate = FixedRateGate(40.0)
    gate.arm(1_000_000_000)
    gate.consume(1_005_000_000)

    assert not gate.ready(1_029_999_999)
    assert gate.ready(1_030_000_000)
