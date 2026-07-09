from pathlib import Path

import numpy as np
from note_seq import midi_io, music_pb2

from ala_pianist.envs import OneHandRoboPianistEnv


def _create_test_midi(path: Path) -> None:
    seq = music_pb2.NoteSequence()
    seq.tempos.add(qpm=90)
    for pitch, start, end in [
        (69, 0.00, 0.35),
        (73, 0.40, 0.75),
    ]:
        seq.notes.add(pitch=pitch, start_time=start, end_time=end, velocity=80, part=0)
    seq.total_time = 0.80
    midi_io.note_sequence_to_midi_file(seq, str(path))


def test_one_hand_env_zero_step(tmp_path):
    midi_path = tmp_path / "tiny.mid"
    _create_test_midi(midi_path)

    env = OneHandRoboPianistEnv(midi_path)
    spec = env.action_spec()

    assert spec.shape == (22,)
    assert "sustain" not in spec.name

    timestep = env.reset()
    assert timestep.observation

    action = np.zeros(spec.shape, dtype=spec.dtype)
    timestep = env.step(action)

    assert timestep is not None
    assert isinstance(env.current_target_keys(), list)
