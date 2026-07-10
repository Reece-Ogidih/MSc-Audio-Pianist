import numpy as np

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.learning.cem import (
    generate_cem_candidates,
    run_single_note_cem,
    write_single_note_cem_midi,
)


def test_cem_candidate_generation_is_seeded_and_bounded(tmp_path):
    midi_path = tmp_path / "cem.mid"
    write_single_note_cem_midi(midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    spec = env.action_spec()
    low = np.asarray(spec.minimum, dtype=float)
    high = np.asarray(spec.maximum, dtype=float)
    mean = (low + high) / 2.0
    std = 0.1 * (high - low)

    first = generate_cem_candidates(
        mean=mean,
        std=std,
        low=low,
        high=high,
        count=4,
        rng=np.random.default_rng(123),
    )
    second = generate_cem_candidates(
        mean=mean,
        std=std,
        low=low,
        high=high,
        count=4,
        rng=np.random.default_rng(123),
    )

    assert len(first) == 4
    for left, right in zip(first, second):
        assert np.allclose(left, right)
        assert left.shape == (22,)
        assert np.all(left >= low)
        assert np.all(left <= high)


def test_single_note_cem_smoke(tmp_path):
    midi_path = tmp_path / "cem.mid"
    write_single_note_cem_midi(midi_path)

    summary = run_single_note_cem(
        midi_path,
        iterations=1,
        candidate_count=3,
        elite_fraction=1 / 3,
        horizon_steps=6,
        seed=5,
        random_search_seed=7,
        random_search_candidates=3,
    )

    assert summary.iterations == 1
    assert summary.candidate_count == 3
    assert summary.best_result.metrics.max_target_key_state >= 0.0
    assert summary.best_result.metrics.max_unintended_key_state >= 0.0
    assert summary.best_result.metrics.native_reward_sum >= 0.0
    assert summary.best_result.metrics.outcome
    assert summary.iteration_summaries
