#!/usr/bin/env python
"""Validate the frozen five-note symbolic controller checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/home/reece_dev/msc-audio-pianist")
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_five_note_factorial_checkpoint_sweep import evaluate_checkpoint  # noqa: E402
from evaluate_general_one_hand_policy import reward_config_from_profile  # noqa: E402
from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv  # noqa: E402


DEFAULT_RELEASE_DIR = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1"
NUMERIC_TOLERANCE = 1e-6


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def smoke_load(release_dir: Path) -> dict[str, Any]:
    manifest = load_json(release_dir / "manifest.json")
    checkpoint = release_dir / manifest["frozen_checkpoint_relative_path"]
    policy = DroQPolicy.load(checkpoint, device="cpu")
    training_config = load_json(release_dir / "provenance/resolved_training_config.json")
    reward_config = reward_config_from_profile(training_config["reward_profile"])
    env = GeneralOneHandGoalEnv(
        curriculum=training_config["curriculum"],
        midi_min=training_config["midi_min"],
        midi_max=training_config["midi_max"],
        sequence_pitches=tuple(tuple(seq) for seq in training_config["sequence_pitches"]),
        sequence_sampling_weights=tuple(training_config["sequence_sampling_weights"]),
        sequence_timing_profile=training_config["sequence_timing_profile"],
        seed=training_config["seed"],
        lookahead=training_config["lookahead"],
        horizon_steps=64,
        action_mode=training_config["action_mode"],
        action_repeat=training_config["action_repeat"],
        reward_config=reward_config,
    )
    obs, _ = env.reset(seed=training_config["seed"])
    action, _ = policy.predict(obs, deterministic=True)
    action = np.asarray(action, dtype=np.float32)
    finite = bool(np.all(np.isfinite(action)))
    steps = 0
    terminated = False
    truncated = False
    while not (terminated or truncated) and steps < env.horizon_steps:
        obs, _, terminated, truncated, _ = env.step(action)
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if not np.all(np.isfinite(action)):
            finite = False
        steps += 1
    return {
        "checkpoint": str(checkpoint),
        "observation_dim": int(env.observation_space.shape[0]),
        "action_dim": int(env.action_space.shape[0]),
        "manifest_observation_dim": int(manifest["observation_dim"]),
        "manifest_action_dim": int(manifest["action_dim"]),
        "finite_deterministic_actions": finite,
        "episode_steps": steps,
        "episode_completed": bool(terminated or truncated),
        "dims_match_manifest": (
            int(env.observation_space.shape[0]) == int(manifest["observation_dim"])
            and int(env.action_space.shape[0]) == int(manifest["action_dim"])
        ),
    }


def compare_rows(local_rows: list[dict[str, Any]], original_csv: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = pd.read_csv(original_csv)
    original = original[original["checkpoint_step"] == 800000].copy()
    local = pd.DataFrame(local_rows)
    merge_keys = ["sequence_name", "sequence_group"]
    merged = local.merge(original, on=merge_keys, suffixes=("_local", "_original"))
    numeric_columns = [
        "pressed_key_precision",
        "pressed_key_recall",
        "pressed_key_f1",
        "timestep_precision",
        "timestep_recall",
        "timestep_f1",
        "incorrect_threshold_crossing_count",
        "missed_target_count",
        "max_unintended_key_state",
        "integrated_unintended_travel",
        "timesteps_above_unintended_soft_threshold",
        "timesteps_above_press_threshold",
        "neighbouring_key_event_count",
        "unrelated_key_event_count",
        "previous_target_late_release_duration",
        "future_target_early_activation_duration",
        "shaped_return",
        "native_reward_sum",
        "deterministic_action_saturation",
    ]
    diffs: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for _, row in merged.iterrows():
        for column in numeric_columns:
            local_value = float(row[f"{column}_local"])
            original_value = float(row[f"{column}_original"])
            diff = local_value - original_value
            abs_diff = abs(diff)
            max_abs_diff = max(max_abs_diff, abs_diff)
            diffs.append(
                {
                    "sequence_name": row["sequence_name"],
                    "sequence_group": row["sequence_group"],
                    "metric": column,
                    "local": local_value,
                    "original": original_value,
                    "difference": diff,
                    "absolute_difference": abs_diff,
                    "within_tolerance": abs_diff <= NUMERIC_TOLERANCE,
                }
            )
    exact_columns = ["observed_pressed_key_set", "strict_outcome", "expected_pressed_key_set"]
    exact_mismatches = []
    for _, row in merged.iterrows():
        for column in exact_columns:
            local_value = row[f"{column}_local"]
            original_value = row[f"{column}_original"]
            if str(local_value) != str(original_value):
                exact_mismatches.append(
                    {
                        "sequence_name": row["sequence_name"],
                        "metric": column,
                        "local": local_value,
                        "original": original_value,
                    }
                )
    summary = {
        "sequence_count_local": int(len(local)),
        "sequence_count_original": int(len(original)),
        "sequence_count_compared": int(len(merged)),
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "max_absolute_numeric_difference": max_abs_diff,
        "numeric_pass": bool(max_abs_diff <= NUMERIC_TOLERANCE and all(math.isfinite(d["absolute_difference"]) for d in diffs)),
        "exact_mismatch_count": len(exact_mismatches),
        "exact_mismatches": exact_mismatches,
        "passed": bool(max_abs_diff <= NUMERIC_TOLERANCE and not exact_mismatches and len(local) == len(original) == len(merged)),
    }
    return diffs, summary


def validate_release(release_dir: Path) -> dict[str, Any]:
    manifest = load_json(release_dir / "manifest.json")
    checkpoint = release_dir / manifest["frozen_checkpoint_relative_path"]
    config = load_json(release_dir / "provenance/resolved_training_config.json")
    validation_dir = release_dir / "validation"
    eval_dir = validation_dir / "reproducibility_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows, reward_rows, health_rows = evaluate_checkpoint(checkpoint, {**config, "algorithm": "droq"}, eval_dir)
    pd.DataFrame(rows).to_csv(eval_dir / "per_checkpoint_per_sequence_metrics.csv", index=False)
    pd.DataFrame(reward_rows).to_csv(eval_dir / "reward_component_summary.csv", index=False)
    pd.DataFrame(health_rows).to_csv(eval_dir / "per_checkpoint_training_health.csv", index=False)
    original_csv = Path(manifest["source_evaluation_paths"]["per_sequence_metrics"])
    diffs, comparison = compare_rows(rows, original_csv)
    smoke = smoke_load(release_dir)
    comparison["smoke_load"] = smoke
    comparison["checkpoint"] = str(checkpoint)
    comparison["original_per_sequence_metrics"] = str(original_csv)
    write_json(validation_dir / "reproducibility_comparison.json", comparison)
    write_csv(validation_dir / "reproducibility_comparison.csv", diffs)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    args = parser.parse_args()
    result = validate_release(args.release_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
