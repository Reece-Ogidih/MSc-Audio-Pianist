import numpy as np

from ala_pianist.rl import GeneralOneHandGoalEnv, GeneralRewardConfig


def test_general_reward_config_fields():
    config = GeneralRewardConfig(
        target_travel_weight=3.0,
        wrong_travel_weight=4.0,
        action_weight=0.01,
    )

    assert config.target_travel_weight == 3.0
    assert config.wrong_travel_weight == 4.0
    assert config.action_weight == 0.01


def test_general_policy_eval_fields_without_trained_model(tmp_path):
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_min=73,
        midi_max=75,
        lookahead=1,
        horizon_steps=1,
    )
    _, info = env.reset()
    _, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert np.isfinite(reward)
    assert terminated or truncated
    assert set(
        [
            "target_keys",
            "pressed_keys",
            "target_key_state",
            "max_unintended_key_state",
            "native_reward",
            "shaped_reward",
            "reward_components",
            "trajectory_quality",
        ]
    ).issubset(info)
    assert info["trajectory_quality"] != "gold_demo_candidate" or info["max_unintended_key_state"] < 0.25
