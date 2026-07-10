import numpy as np

from ala_pianist.controllers.action_library import KEYSET_MIDI
from ala_pianist.rl import KeysetPianoGymEnv


def test_keyset_rl_env_reset_and_step(tmp_path):
    env = KeysetPianoGymEnv(midi_dir=tmp_path, horizon_steps=2)
    obs, info = env.reset(seed=123)

    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (22,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    assert info["target_midi"] in KEYSET_MIDI
    assert "sustain" not in env.wrapped_env.action_names()

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["sustain_state"] == 0.0


def test_keyset_rl_env_can_force_target(tmp_path):
    env = KeysetPianoGymEnv(midi_dir=tmp_path, horizon_steps=1)
    obs, info = env.reset(options={"target_midi": 73})

    assert env.observation_space.contains(obs)
    assert info["target_midi"] == 73
    assert info["target_key"] == 52
