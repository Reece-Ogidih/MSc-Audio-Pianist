import numpy as np

from ala_pianist.rl import GeneralOneHandGoalEnv
from ala_pianist.rl.general_env_diagnostics import (
    D_SHARP_5_KEY,
    goal_timing_diagnostic,
    reward_breakdown,
    run_dsharp5_residual_policy_diagnostic,
    run_known_dsharp5_base_diagnostic,
    run_zero_action_diagnostic,
    write_dsharp5_diagnostic_midi,
)


def test_goal_timing_diagnostic_reports_expected_fields(tmp_path):
    midi_path = write_dsharp5_diagnostic_midi(tmp_path / "dsharp5.mid")
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=73,
        midi_max=75,
        lookahead=1,
        horizon_steps=4,
    )

    result = goal_timing_diagnostic(env, steps=2)

    assert result["native_goal_shape"] == (178,)
    assert result["expected_target_key"] == D_SHARP_5_KEY
    assert "target_key_seen_in_goal" in result
    assert result["records"]
    assert "current_target_keys" in result["records"][0]
    assert "future_target_keys" in result["records"][0]


def test_reward_breakdown_contains_named_components(tmp_path):
    midi_path = write_dsharp5_diagnostic_midi(tmp_path / "dsharp5.mid")
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=73,
        midi_max=75,
        lookahead=1,
        horizon_steps=1,
    )
    _, info = env.reset()
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    breakdown = reward_breakdown(env, info)

    assert np.isfinite(reward)
    assert {
        "target_travel_reward",
        "wrong_key_penalty",
        "wrong_pressed_key_penalty",
        "action_magnitude_penalty",
        "smoothness_penalty",
        "fingering_reward",
        "native_reward_component",
        "total_shaped_reward",
    }.issubset(breakdown)


def test_zero_action_diagnostic_runs_without_learning(tmp_path):
    midi_path = write_dsharp5_diagnostic_midi(tmp_path / "dsharp5.mid")
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=73,
        midi_max=75,
        lookahead=1,
        horizon_steps=2,
    )

    result = run_zero_action_diagnostic(env, horizon_steps=2)

    assert result.name == "zero_action"
    assert result.steps
    assert "target_travel_reward" in result.final_reward_breakdown
    assert result.skipped_reason is None


def test_known_action_diagnostic_runs_or_skips_cleanly(tmp_path):
    midi_path = write_dsharp5_diagnostic_midi(tmp_path / "dsharp5.mid")
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=73,
        midi_max=75,
        lookahead=1,
        horizon_steps=1,
    )

    result = run_known_dsharp5_base_diagnostic(env, horizon_steps=1)

    assert result.name.startswith("known_dsharp5_base_action")
    assert result.skipped_reason is not None or "target_travel_reward" in result.final_reward_breakdown


def test_residual_policy_diagnostic_skips_cleanly_when_model_missing(tmp_path):
    midi_path = write_dsharp5_diagnostic_midi(tmp_path / "dsharp5.mid")
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=73,
        midi_max=75,
        lookahead=1,
        horizon_steps=1,
    )

    result = run_dsharp5_residual_policy_diagnostic(
        env,
        model_path=tmp_path / "missing_model.zip",
        horizon_steps=1,
    )

    assert result.name == "known_dsharp5_residual_policy"
    assert result.skipped_reason is not None
