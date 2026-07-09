import numpy as np

from ala_pianist.baselines import ScriptedDiagnosticController, run_scripted_diagnostic
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


def test_scripted_controller_produces_public_action(tmp_path):
    midi_path = tmp_path / "scripted.mid"
    write_monophonic_midi([NoteEvent(69, 0.0, 0.2, 80)], midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()

    controller = ScriptedDiagnosticController()
    action, active_names, selected_key, selected_finger = controller.action(env)
    spec = env.action_spec()

    assert action.shape == (22,)
    assert action.dtype == spec.dtype
    assert np.all(action >= spec.minimum)
    assert np.all(action <= spec.maximum)
    assert selected_key == 69 - 21
    assert selected_finger is not None
    assert "forearm_tx" in active_names
    assert "forearm_ty" in active_names
    assert "sustain" not in env.action_names()


def test_scripted_diagnostic_rollout_logs_generated_midi(tmp_path):
    midi_path = tmp_path / "rollout.mid"
    write_monophonic_midi(
        [
            NoteEvent(69, 0.00, 0.20, 80),
            NoteEvent(73, 0.25, 0.20, 80),
            NoteEvent(74, 0.50, 0.20, 80),
        ],
        midi_path,
    )
    env = ALAOneHandEnv(midi_path)
    logs = run_scripted_diagnostic(
        env,
        ScriptedDiagnosticController(),
        max_steps=32,
    )

    assert logs
    assert any(log.target_keys == (69 - 21,) for log in logs)
    assert any(log.target_keys == (73 - 21,) for log in logs)
    assert all(log.status for log in logs)
    target_logs = [log for log in logs if log.selected_key is not None]
    assert target_logs
    assert all(log.selected_note for log in target_logs)
    assert all(log.target_key_state is not None for log in target_logs)
    assert all(log.max_unintended_key_state >= 0.0 for log in logs)
    assert any(log.nearest_fingertip_distance is not None for log in target_logs)
    assert all(isinstance(log.target_contact, bool) for log in logs)
    assert all(isinstance(log.any_key_contact, bool) for log in logs)
    assert all(log.diagnostic_category for log in logs)
    assert env.task.piano.sustain_state[0] == 0.0


def test_env_read_only_diagnostic_helpers(tmp_path):
    midi_path = tmp_path / "diagnostics.mid"
    write_monophonic_midi([NoteEvent(73, 0.0, 0.2, 80)], midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()

    target_key = 73 - 21
    key_states = env.piano_key_states()
    nearest = env.nearest_fingertip_to_key(target_key)

    assert key_states.shape == (88,)
    assert env.target_key_state(target_key) is not None
    assert env.max_unintended_key_state(target_key) >= 0.0
    assert nearest is not None
    assert nearest["fingertip"]
    assert nearest["distance"] >= 0.0
    assert env.key_press_region_position(target_key).shape == (3,)
    assert isinstance(env.contact_pairs(), list)
    assert env.note_name_for_key(target_key)
