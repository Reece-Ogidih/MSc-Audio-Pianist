import numpy as np

from ala_pianist.controllers.rl_policy_controller import RLPolicyController
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


def test_rl_policy_controller_zero_fallback(tmp_path):
    midi_path = tmp_path / "note.mid"
    write_monophonic_midi([NoteEvent(75, 0.0, 1.0, 90)], midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    controller = RLPolicyController(model_path=None)

    action = controller.action(env, target_midi=75, step_count=0)
    assert action.shape == (22,)
    assert np.allclose(action, 0.0)


def test_rl_policy_controller_action_can_step_env(tmp_path):
    midi_path = tmp_path / "note.mid"
    write_monophonic_midi([NoteEvent(69, 0.0, 1.0, 90)], midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    controller = RLPolicyController(model_path=None)
    timestep = env.step(controller.action(env, target_midi=69, step_count=0))

    assert timestep is not None
    assert env.task.piano.sustain_state[0] == 0.0
