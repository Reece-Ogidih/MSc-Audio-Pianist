import importlib.util
from pathlib import Path

import numpy as np

from ala_pianist.rl import GeneralOneHandGoalEnv, GeneralRewardConfig


def _training_script_module():
    script_path = Path("/home/reece_dev/msc-audio-pianist/scripts/train_general_one_hand_policy.py")
    spec = importlib.util.spec_from_file_location("train_general_one_hand_policy", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_general_reward_config_fields():
    config = GeneralRewardConfig(
        target_travel_weight=3.0,
        wrong_travel_weight=4.0,
        action_weight=0.01,
    )

    assert config.target_travel_weight == 3.0
    assert config.wrong_travel_weight == 4.0
    assert config.action_weight == 0.01


def test_training_arg_helpers_parse_midi_pitches_and_booleans():
    module = _training_script_module()
    assert module.parse_midi_pitches("73,75") == (73, 75)
    assert module.parse_midi_pitches("73, 74,75") == (73, 74, 75)
    assert module.parse_midi_pitches(None) is None
    assert module.parse_bool("false") is False
    assert module.parse_bool("true") is True


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
