import numpy as np

from ala_pianist.controllers import HybridPipeline1Controller
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


def test_hybrid_controller_falls_back_without_model(tmp_path):
    midi_path = tmp_path / "note.mid"
    write_monophonic_midi([NoteEvent(69, 0.0, 1.0, 90)], midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    controller = HybridPipeline1Controller(None)
    fallback = np.ones(env.action_spec().shape, dtype=env.action_spec().dtype)

    action0 = controller.action(env, target_midi=69, fallback_action=fallback, step_count=0)
    action5 = controller.action(env, target_midi=69, fallback_action=fallback, step_count=5)

    assert action0.shape == (22,)
    assert np.allclose(action0, fallback / 6.0)
    assert np.allclose(action5, fallback)
