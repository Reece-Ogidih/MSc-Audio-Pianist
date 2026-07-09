import numpy as np

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.learning.random_search import (
    generate_random_candidates,
    run_single_note_random_search,
    write_single_note_learning_midi,
)


def test_random_candidates_are_seeded_and_bounded(tmp_path):
    midi_path = tmp_path / "note.mid"
    write_single_note_learning_midi(midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()

    first = generate_random_candidates(env, count=3, seed=123)
    second = generate_random_candidates(env, count=3, seed=123)
    spec = env.action_spec()

    assert first == second
    assert len(first) == 3
    for candidate in first:
        action = np.asarray(candidate.action)
        assert action.shape == (22,)
        assert np.all(action >= spec.minimum)
        assert np.all(action <= spec.maximum)


def test_single_note_random_search_smoke(tmp_path):
    midi_path = tmp_path / "note.mid"
    output_path = tmp_path / "summary.json"
    write_single_note_learning_midi(midi_path)

    summary = run_single_note_random_search(
        midi_path,
        candidate_count=2,
        horizon_steps=6,
        seed=5,
        output_path=output_path,
    )

    assert summary.candidate_count == 2
    assert len(summary.results) == 2
    assert summary.zero_result.max_target_key_state >= 0.0
    assert summary.scripted_result.max_target_key_state >= 0.0
    assert summary.best_result.max_target_key_state >= 0.0
    assert summary.best_result.outcome
    assert output_path.exists()
