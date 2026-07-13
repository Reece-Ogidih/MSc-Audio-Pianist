import numpy as np

from ala_pianist.rl import ResidualSingleNoteEnv, get_dirty_dsharp5_base_action
from ala_pianist.rl.residual_env import evaluate_residual_action


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
    for mode in ("target_travel_first", "cleanliness", "constrained_cleanliness"):
        env = ResidualSingleNoteEnv(midi_path=tmp_path / f"{mode}.mid", reward_mode=mode)
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert info["reward_mode"] == mode


def test_residual_env_supports_configurable_csharp5_target(tmp_path):
    env = ResidualSingleNoteEnv(
        midi_path=tmp_path / "csharp5.mid",
        target_midi=73,
        wrong_key=54,
        wrong_keys=(54, 55, 56),
        reward_mode="cleanliness",
        residual_scale=0.03,
        base_action_penalty=0.05,
        base_action=np.zeros(22, dtype=np.float32),
    )
    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert info["target_midi"] == 73
    assert info["target_key"] == 52
    assert info["wrong_key"] == 54
    assert info["wrong_keys"] == (54, 55, 56)
    assert info["base_action_penalty_weight"] == 0.05
    assert env.residual_scale == 0.03
    assert info["sustain_state"] == 0.0

    obs, reward, _, _, info = env.step(np.zeros(22, dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert "wrong_key_state" in info
    assert "key_54_state" in info
    assert "key_53_state" in info
    assert "key_55_state" in info
    assert "key_56_state" in info
    assert "residual_magnitude" in info
    assert "action_deviation_from_base" in info
    assert info["sustain_state"] == 0.0


def test_residual_evaluation_reports_csharp5_wrong_key_fields(tmp_path):
    result = evaluate_residual_action(
        target_midi=73,
        wrong_key=54,
        wrong_keys=(54, 55, 56),
        base_action=np.zeros(22, dtype=np.float32),
        residual_scale=0.05,
        reward_mode="constrained_cleanliness",
        base_action_penalty=0.1,
        horizon_steps=1,
    )

    assert result["target_key"] == 52
    assert result["wrong_keys"] == (54, 55, 56)
    assert "max_key_52_state" in result
    assert "max_key_53_state" in result
    assert "max_key_54_state" in result
    assert "max_key_55_state" in result
    assert "max_key_56_state" in result
    assert "max_residual_magnitude" in result
    assert "max_action_deviation_from_base" in result


def test_residual_env_supports_d5_target_with_neighbour_penalties(tmp_path):
    env = ResidualSingleNoteEnv(
        midi_path=tmp_path / "d5.mid",
        target_midi=74,
        wrong_key=54,
        wrong_keys=(52, 54, 55, 56),
        reward_mode="constrained_cleanliness",
        residual_scale=0.05,
        base_action_penalty=0.1,
        base_action=np.zeros(22, dtype=np.float32),
    )
    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert info["target_midi"] == 74
    assert info["target_key"] == 53
    assert info["wrong_keys"] == (52, 54, 55, 56)

    obs, reward, _, _, info = env.step(np.zeros(22, dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert "key_53_state" in info


def test_dirty_dsharp5_base_action_available(tmp_path):
    action = get_dirty_dsharp5_base_action(tmp_path / "base.mid")
    assert action.shape == (22,)
    assert np.all(np.isfinite(action))
