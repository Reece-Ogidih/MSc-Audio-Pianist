import numpy as np

from ala_pianist.controllers import HybridPipeline1Controller
from ala_pianist.controllers.hybrid_controller import ResidualPolicy
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi
from ala_pianist.rl.residual_env import ResidualSingleNoteEnv


class _FakeModel:
    def predict(self, obs, deterministic=True):
        del obs, deterministic
        return np.zeros(22, dtype=np.float32), None


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


def test_hybrid_controller_routes_configured_residual_policy(tmp_path):
    midi_path = tmp_path / "csharp5.mid"
    write_monophonic_midi([NoteEvent(73, 0.0, 1.0, 90)], midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    helper = ResidualSingleNoteEnv(
        midi_path=tmp_path / "helper.mid",
        target_midi=73,
        wrong_key=54,
        base_action=np.zeros(22, dtype=np.float32),
    )
    controller = HybridPipeline1Controller(None)
    controller.policies[73] = ResidualPolicy(
        target_midi=73,
        model_path=tmp_path / "fake.zip",
        model=_FakeModel(),
        helper_env=helper,
    )
    fallback = np.ones(env.action_spec().shape, dtype=env.action_spec().dtype)

    action = controller.action(env, target_midi=73, fallback_action=fallback, step_count=5)

    assert action.shape == (22,)
    assert not np.allclose(action, fallback)


def test_hybrid_controller_can_route_dsharp5_and_d5_policies(tmp_path):
    midi_path = tmp_path / "d5.mid"
    write_monophonic_midi([NoteEvent(74, 0.0, 1.0, 90)], midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    controller = HybridPipeline1Controller(None)
    for midi in (75, 74):
        helper = ResidualSingleNoteEnv(
            midi_path=tmp_path / f"helper_{midi}.mid",
            target_midi=midi,
            wrong_key=54 if midi == 74 else 50,
            base_action=np.zeros(22, dtype=np.float32),
        )
        controller.policies[midi] = ResidualPolicy(
            target_midi=midi,
            model_path=tmp_path / f"fake_{midi}.zip",
            model=_FakeModel(),
            helper_env=helper,
        )

    fallback = np.ones(env.action_spec().shape, dtype=env.action_spec().dtype)
    action = controller.action(env, target_midi=74, fallback_action=fallback, step_count=5)

    assert set(controller.policies) == {74, 75}
    assert action.shape == (22,)
    assert not np.allclose(action, fallback)
