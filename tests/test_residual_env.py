import numpy as np

from ala_pianist.rl import ResidualSingleNoteEnv, get_dirty_dsharp5_base_action


def test_residual_env_reset_and_step(tmp_path):
    env = ResidualSingleNoteEnv(midi_path=tmp_path / "note.mid", reward_mode="target_travel_first")
    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (22,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    assert len(env.base_action) == 22
    assert "sustain" not in env.wrapped_env.action_names()
    assert info["sustain_state"] == 0.0

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["sustain_state"] == 0.0


def test_residual_env_reward_modes_supported(tmp_path):
    for mode in ("target_travel_first", "cleanliness"):
        env = ResidualSingleNoteEnv(midi_path=tmp_path / f"{mode}.mid", reward_mode=mode)
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert info["reward_mode"] == mode


def test_dirty_dsharp5_base_action_available(tmp_path):
    action = get_dirty_dsharp5_base_action(tmp_path / "base.mid")
    assert action.shape == (22,)
    assert np.all(np.isfinite(action))
