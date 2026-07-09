from pathlib import Path

import numpy as np

from ala_pianist.envs import ALAOneHandEnv, OneHandRoboPianistEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi
from robopianist.music import midi_file


def _create_test_midi(path: Path) -> None:
    write_monophonic_midi(
        [
            NoteEvent(69, 0.00, 0.20, 80),
            NoteEvent(73, 0.25, 0.20, 80),
            NoteEvent(74, 0.50, 0.20, 80),
            NoteEvent(75, 0.75, 0.20, 80),
        ],
        path,
        title="test one hand env",
    )


def test_one_hand_env_zero_step(tmp_path):
    midi_path = tmp_path / "tiny.mid"
    _create_test_midi(midi_path)

    env = ALAOneHandEnv(midi_path)
    spec = env.action_spec()

    assert spec.shape == (22,)
    assert "sustain" not in spec.name

    timestep = env.reset()
    assert timestep.observation

    action = np.zeros(spec.shape, dtype=spec.dtype)
    timestep = env.step(action)

    assert timestep is not None
    assert isinstance(env.current_target_keys(), list)
    assert env.task.piano.sustain_state[0] == 0.0


def test_generated_midi_target_timing(tmp_path):
    midi_path = tmp_path / "timing.mid"
    events = [
        NoteEvent(69, 0.00, 0.20, 80),
        NoteEvent(73, 0.25, 0.20, 80),
        NoteEvent(74, 0.50, 0.20, 80),
        NoteEvent(75, 0.75, 0.20, 80),
    ]
    write_monophonic_midi(events, midi_path)

    for event in events:
        assert midi_file.midi_number_to_key_number(event.pitch) == event.pitch - 21

    env = OneHandRoboPianistEnv(midi_path)
    timestep = env.reset()
    zero = np.zeros(env.action_spec().shape, dtype=env.action_spec().dtype)

    observed = []
    expected = []
    for _ in range(20):
        observed.append(env.current_target_keys())
        notes = env.task._notes[env.task._t_idx] if env.task._t_idx < len(env.task._notes) else []
        expected.append([int(note.key) for note in notes])
        timestep = env.step(zero)
        assert env.task.piano.sustain_state[0] == 0.0

    assert observed == expected
    assert [69 - 21] in observed
    assert [73 - 21] in observed
    assert timestep is not None
