import importlib.util
from pathlib import Path

import numpy as np

from ala_pianist.evaluation.unintended import (
    classify_unintended_key,
    unintended_penalty_components,
)
from ala_pianist.rl import GeneralRewardConfig


def _load_train_module():
    path = Path("/home/reece_dev/msc-audio-pianist/scripts/train_general_one_hand_policy.py")
    spec = importlib.util.spec_from_file_location("train_general_one_hand_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_unrelated_previous_future_and_neighbour_classification():
    assert classify_unintended_key(
        51,
        value=0.8,
        current_target_keys={52},
        previous_target_keys=set(),
        future_target_keys=set(),
        press_threshold=0.5,
    ).category == "neighbouring_key_displacement"
    assert classify_unintended_key(
        54,
        value=0.8,
        current_target_keys={52},
        previous_target_keys={54},
        future_target_keys=set(),
        press_threshold=0.5,
    ).category == "previous_note_late_release"
    assert classify_unintended_key(
        54,
        value=0.8,
        current_target_keys={52},
        previous_target_keys=set(),
        future_target_keys={54},
        press_threshold=0.5,
    ).category == "future_note_early_activation"
    assert classify_unintended_key(
        60,
        value=0.8,
        current_target_keys={52},
        previous_target_keys=set(),
        future_target_keys=set(),
        press_threshold=0.5,
    ).category == "unrelated_key_activation"


def test_sensitive_penalties_ignore_correct_active_target():
    states = np.zeros(88, dtype=float)
    states[52] = 1.0
    components = unintended_penalty_components(
        states,
        current_target_keys={52},
        previous_target_keys=set(),
        future_target_keys=set(),
        soft_threshold=0.2,
        press_threshold=0.5,
    )
    assert components["unintended_continuous_travel"] == 0.0
    assert components["unintended_pressed_event_count"] == 0.0
    assert components["unintended_integrated_duration"] == 0.0


def test_soft_near_press_late_and_early_penalties():
    states = np.zeros(88, dtype=float)
    states[60] = 0.4
    states[54] = 0.8
    states[53] = 0.7
    components = unintended_penalty_components(
        states,
        current_target_keys={52},
        previous_target_keys={54},
        future_target_keys={53},
        soft_threshold=0.2,
        press_threshold=0.5,
    )
    assert components["unintended_continuous_travel"] > 0.0
    assert components["unintended_near_press_barrier"] > 0.0
    assert components["late_release_travel"] > 0.0
    assert components["early_activation_travel"] > 0.0
    assert components["unintended_integrated_duration"] > components["unintended_continuous_travel"]


def test_transition_cleanup_profile_is_unchanged_by_sensitive_profile():
    train_module = _load_train_module()
    baseline = train_module.reward_config_from_profile("transition_cleanup")
    sensitive = train_module.reward_config_from_profile("transition_cleanup_sensitive_v1")
    assert isinstance(sensitive, GeneralRewardConfig)
    assert baseline.unintended_travel_weight == 0.0
    assert baseline.duration_weight == 0.0
    assert sensitive.unintended_travel_weight > 0.0
    assert sensitive.duration_weight > 0.0
