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
    assert module.parse_pitch_sampling_weights("0.7,0.3") == (0.7, 0.3)
    assert module.parse_sequence_pitches("73;75;73,75;75,73") == (
        (73,),
        (75,),
        (73, 75),
        (75, 73),
    )
    assert module.parse_midi_pitches(None) is None
    assert module.parse_bool("false") is False
    assert module.parse_bool("true") is True


def test_training_sequence_timing_dry_run_args_exist(tmp_path):
    module = _training_script_module()

    assert module.parse_sequence_pitches("73,75") == ((73, 75),)
    assert module.parse_pitch_sampling_weights("0.25,0.75") == (0.25, 0.75)


def test_reward_profile_press_bonus_exists():
    module = _training_script_module()
    config = module.reward_config_from_profile("press_bonus")

    assert config.target_activation_bonus > 0.0
    assert config.target_activation_threshold == 0.9
    assert config.high_unintended_weight > 0.0


def test_reward_profile_cleanup_exists_and_is_gated():
    module = _training_script_module()
    config = module.reward_config_from_profile("cleanup")

    assert config.cleanup_gate_threshold < 1.0
    assert config.gated_unintended_weight > 0.0
    assert config.gated_wrong_pressed_weight > 0.0
    assert config.nearby_wrong_key_weight > 0.0
    assert config.wrong_travel_weight < config.gated_unintended_weight


def test_reward_profile_anti_coupling_exists_and_is_asymmetric():
    module = _training_script_module()
    config = module.reward_config_from_profile("anti_coupling")

    assert config.cleanup_gate_threshold < 1.0
    assert config.csharp_dsharp_key54_weight > config.dsharp_csharp_key52_weight
    assert config.csharp_dsharp_pressed_weight > config.dsharp_csharp_pressed_weight
    assert config.action_weight < 0.01


def test_reward_profile_transition_cleanup_exists():
    module = _training_script_module()
    config = module.reward_config_from_profile("transition_cleanup")

    assert config.release_previous_key_weight > 0.0
    assert config.transition_stray_key_weight > 0.0
    assert config.transition_stray_pressed_weight > 0.0
    assert config.csharp_dsharp_key54_weight > config.dsharp_csharp_key52_weight


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
