import numpy as np

from ala_pianist.rl import SingleNotePianoGymEnv, write_single_note_rl_midi


def test_single_note_gym_env_reset_and_step(tmp_path):
    midi_path = tmp_path / "single_note.mid"
    write_single_note_rl_midi(midi_path)
    env = SingleNotePianoGymEnv(midi_path, horizon_steps=2)

    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (22,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    assert "sustain" not in env.wrapped_env.action_names()
    assert info["sustain_state"] == 0.0

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["sustain_state"] == 0.0
    assert "target_key_state" in info
    assert "native_reward" in info
    assert info["action_space_mode"] == "normalized_minus_one_to_one"


def test_single_note_gym_env_truncates_at_horizon(tmp_path):
    midi_path = tmp_path / "single_note.mid"
    write_single_note_rl_midi(midi_path)
    env = SingleNotePianoGymEnv(midi_path, horizon_steps=1)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    )

    assert env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert terminated or truncated
    assert info["sustain_state"] == 0.0


def test_single_note_gym_env_rescales_normalized_actions(tmp_path):
    midi_path = tmp_path / "single_note.mid"
    write_single_note_rl_midi(midi_path)
    env = SingleNotePianoGymEnv(midi_path, horizon_steps=2)
    native_low, native_high = env.native_action_bounds

    assert np.allclose(env.rescale_action(np.full(22, -1.0, dtype=np.float32)), native_low)
    assert np.allclose(env.rescale_action(np.full(22, 1.0, dtype=np.float32)), native_high)
    midpoint = env.rescale_action(np.zeros(22, dtype=np.float32))
    assert np.all(midpoint >= native_low)
    assert np.all(midpoint <= native_high)
