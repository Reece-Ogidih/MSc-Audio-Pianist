from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv


ROOT = Path("/home/reece_dev/msc-audio-pianist")
FROZEN_CKPT = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt"


def _training_module():
    path = ROOT / "scripts/train_general_one_hand_policy.py"
    spec = importlib.util.spec_from_file_location("train_general_one_hand_policy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _prepare_module():
    path = ROOT / "scripts/prepare_refinement_canary.py"
    spec = importlib.util.spec_from_file_location("prepare_refinement_canary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_completion_reward_profiles_are_distinct():
    module = _training_module()
    control = module.reward_config_from_profile("transition_cleanup_sensitive_v1")
    release = module.reward_config_from_profile("transition_cleanup_release_completion_v2")
    motion = module.reward_config_from_profile("transition_cleanup_release_completion_motion_v2")

    assert control.release_completion_release_weight == 0.0
    assert release.release_completion_release_weight > 0.0
    assert release.release_completion_bonus > 0.0
    assert release.transition_action_rate_weight == 0.0
    assert motion.release_completion_release_weight == release.release_completion_release_weight
    assert motion.release_completion_bonus == release.release_completion_bonus
    assert motion.transition_action_rate_weight > 0.0
    assert motion.transition_saturation_weight > 0.0


def test_release_completion_signal_is_one_shot(monkeypatch, tmp_path):
    module = _training_module()
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_min=72,
        midi_max=76,
        lookahead=1,
        horizon_steps=2,
        reward_config=module.reward_config_from_profile("transition_cleanup_release_completion_v2"),
    )
    states = np.zeros(88, dtype=np.float32)
    states[51] = 0.75
    states[52] = 0.10
    env._last_nonempty_target_keys = (52,)
    monkeypatch.setattr(env, "current_target_keys", lambda: [51])
    monkeypatch.setattr(env, "future_target_keys", lambda: [])
    monkeypatch.setattr(env, "piano_key_states", lambda: states.copy())
    monkeypatch.setattr(env, "current_pressed_keys", lambda: [51])

    first = env._reward_components(
        normalized_action=np.zeros(22, dtype=np.float32),
        native_reward=0.0,
    )
    second = env._reward_components(
        normalized_action=np.zeros(22, dtype=np.float32),
        native_reward=0.0,
    )

    assert first["release_completion_transition_gate"] == 1.0
    assert first["release_completion_target_changed_event"] == 1.0
    assert first["release_completion_second_target_event"] == 1.0
    assert first["release_completion_release_achieved_event"] == 1.0
    assert second["release_completion_second_target_event"] == 0.0


def test_transition_motion_penalties_fire_only_during_transition(monkeypatch, tmp_path):
    module = _training_module()
    env = GeneralOneHandGoalEnv(
        generated_midi_dir=tmp_path,
        curriculum="single_notes",
        midi_min=72,
        midi_max=76,
        lookahead=1,
        horizon_steps=2,
        reward_config=module.reward_config_from_profile("transition_cleanup_release_completion_motion_v2"),
    )
    states = np.zeros(88, dtype=np.float32)
    states[51] = 0.4
    states[52] = 0.7
    action = np.ones(22, dtype=np.float32)
    env._previous_normalized_action = -np.ones(22, dtype=np.float32)

    monkeypatch.setattr(env, "future_target_keys", lambda: [])
    monkeypatch.setattr(env, "piano_key_states", lambda: states.copy())
    monkeypatch.setattr(env, "current_pressed_keys", lambda: [])

    monkeypatch.setattr(env, "current_target_keys", lambda: [52])
    env._last_nonempty_target_keys = (52,)
    no_transition = env._reward_components(normalized_action=action, native_reward=0.0)

    monkeypatch.setattr(env, "current_target_keys", lambda: [51])
    env._last_nonempty_target_keys = (52,)
    transition = env._reward_components(normalized_action=action, native_reward=0.0)

    assert no_transition["transition_action_rate"] == 0.0
    assert no_transition["transition_saturation"] == 0.0
    assert transition["transition_action_rate"] > 0.0
    assert transition["transition_saturation"] > 0.0


def test_canary_sampling_weights_are_derived_and_bounded():
    module = _prepare_module()
    weights, derivation = module.derive_weights()

    assert len(weights) == 13
    assert abs(sum(weights) - 1.0) < 1e-8
    assert max(weights) <= 0.12
    assert derivation["transition_scores"]["73;72"] > derivation["transition_scores"]["75;76"]


def test_frozen_checkpoint_loads_for_warm_start():
    policy = DroQPolicy.load(FROZEN_CKPT, device="cpu")
    assert policy.agent.config.action_dim == 22
    assert policy.agent.config.observation_dim == 301

