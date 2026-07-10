import numpy as np

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.evaluation import record_action_rollout, save_trajectory_json
from ala_pianist.music import NoteEvent, write_monophonic_midi


def test_trajectory_records_required_fields(tmp_path):
    midi_path = tmp_path / "note.mid"
    write_monophonic_midi([NoteEvent(75, 0.0, 1.0, 90)], midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    records = record_action_rollout(
        env,
        target_midi=75,
        action=np.zeros(env.action_spec().shape),
        horizon_steps=2,
    )
    path = save_trajectory_json(records, tmp_path / "demo.json")

    assert path.exists()
    assert records
    record = records[0]
    assert record.target_midi == 75
    assert record.target_key == 54
    assert len(record.action) == 22
    assert len(record.compact_observation) == 6
    assert isinstance(record.pressed_keys, tuple)
    assert record.closest_finger_to_target
    assert record.target_contact_finger
    assert record.outcome
