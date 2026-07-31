#!/usr/bin/env python
"""Evaluate refinement-canary arms with behavior and motion-quality metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/home/reece_dev/msc-audio-pianist")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ala_pianist.evaluation import binary_key_vector, pressed_key_metrics, timestep_key_metrics  # noqa: E402
from ala_pianist.evaluation.motion_quality import action_quality  # noqa: E402
from ala_pianist.evaluation.unintended import classify_unintended_keys  # noqa: E402
from ala_pianist.music import assign_right_hand_fingering, sequence_timing_from_profile, write_sequence_midi  # noqa: E402
from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv  # noqa: E402
from train_general_one_hand_policy import reward_config_from_profile  # noqa: E402


FROZEN = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1"
CANARY = FROZEN / "refinement_canary"
FROZEN_CKPT = FROZEN / "checkpoint_800000_steps.pt"
EXPECTED_CHECKPOINT_STEPS = (5000, 10000, 25000, 50000, 75000)
SEQUENCES = {
    "single_72": [72],
    "single_73": [73],
    "single_74": [74],
    "single_75": [75],
    "single_76": [76],
    "trained_72_73": [72, 73],
    "trained_73_72": [73, 72],
    "trained_73_74": [73, 74],
    "trained_74_73": [74, 73],
    "trained_74_75": [74, 75],
    "trained_75_74": [75, 74],
    "trained_75_76": [75, 76],
    "trained_76_75": [76, 75],
}


@dataclass(frozen=True)
class CheckpointSpec:
    arm: str
    reward_profile: str
    step: int
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-root", type=Path, default=CANARY)
    parser.add_argument("--output-dir", type=Path, default=CANARY / "validation")
    parser.add_argument("--baseline-checkpoint", type=Path, default=FROZEN_CKPT)
    parser.add_argument("--include-frozen-baseline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-checkpoint-count", type=int, default=15)
    args = parser.parse_args()
    if args.dry_run:
        result = discovery_payload(
            args.canary_root,
            args.output_dir,
            baseline_checkpoint=args.baseline_checkpoint,
            expected_checkpoint_count=args.expected_checkpoint_count,
        )
    else:
        result = evaluate(
            args.canary_root,
            args.output_dir,
            baseline_checkpoint=args.baseline_checkpoint,
            include_frozen=args.include_frozen_baseline,
            expected_checkpoint_count=args.expected_checkpoint_count,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate(
    canary_root: Path,
    output_dir: Path,
    *,
    baseline_checkpoint: Path,
    include_frozen: bool,
    expected_checkpoint_count: int = 15,
) -> dict:
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(canary_root)
    if len(checkpoints) != expected_checkpoint_count:
        raise ValueError(
            f"Expected {expected_checkpoint_count} canary checkpoints, found {len(checkpoints)}."
        )
    eval_specs = []
    if include_frozen:
        eval_specs.append(
            CheckpointSpec("frozen_800k_baseline", "transition_cleanup_sensitive_v1", 0, baseline_checkpoint)
        )
    eval_specs.extend(checkpoints)
    rows = []
    reward_rows = []
    for spec in eval_specs:
        for sequence_id, pitches in SEQUENCES.items():
            row, components = evaluate_checkpoint(
                spec.path,
                arm_name=spec.arm,
                reward_profile=spec.reward_profile,
                checkpoint_step=spec.step,
                sequence_id=sequence_id,
                pitches=pitches,
                output_dir=output_dir,
            )
            rows.append(row)
            reward_rows.extend(components)
    summary = summarize(rows)
    arm_comparison = compare_to_control(summary)
    ab_delta = compare_arms(summary, "release_completion_v2", "control_continue_sensitive_v1", "a_vs_b_release_completion_delta.csv")
    bc_delta = compare_arms(summary, "release_completion_motion_v2", "release_completion_v2", "b_vs_c_motion_delta.csv")
    baseline_delta = compare_to_baseline(summary)
    recommendation = select_candidate(summary)
    write_csv(output_dir / "per_checkpoint_per_sequence_metrics.csv", rows)
    write_csv(output_dir / "reward_component_summary.csv", reward_rows)
    write_csv(output_dir / "per_checkpoint_summary.csv", summary)
    write_csv(output_dir / "arm_comparison.csv", arm_comparison)
    write_csv(output_dir / "a_vs_b_release_completion_delta.csv", ab_delta)
    write_csv(output_dir / "b_vs_c_motion_delta.csv", bc_delta)
    write_csv(output_dir / "vs_step0_baseline_delta.csv", baseline_delta)
    (output_dir / "selection_recommendation.json").write_text(
        json.dumps(recommendation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = output_dir / "comparison_report.md"
    write_report(report_path, summary, arm_comparison, ab_delta, bc_delta, baseline_delta, recommendation)
    payload = {
        "runtime_seconds": time.time() - start,
        "evaluated_checkpoints": [str(item.path) for item in eval_specs],
        "outputs": {
            "per_checkpoint_per_sequence_metrics": str(output_dir / "per_checkpoint_per_sequence_metrics.csv"),
            "reward_component_summary": str(output_dir / "reward_component_summary.csv"),
            "per_checkpoint_summary": str(output_dir / "per_checkpoint_summary.csv"),
            "arm_comparison": str(output_dir / "arm_comparison.csv"),
            "a_vs_b_release_completion_delta": str(output_dir / "a_vs_b_release_completion_delta.csv"),
            "b_vs_c_motion_delta": str(output_dir / "b_vs_c_motion_delta.csv"),
            "vs_step0_baseline_delta": str(output_dir / "vs_step0_baseline_delta.csv"),
            "comparison_report": str(report_path),
            "selection_recommendation": str(output_dir / "selection_recommendation.json"),
        },
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def discovery_payload(
    canary_root: Path,
    output_dir: Path,
    *,
    baseline_checkpoint: Path,
    expected_checkpoint_count: int,
) -> dict:
    manifest = load_manifest(canary_root)
    checkpoints = discover_checkpoints(canary_root)
    if len(checkpoints) != expected_checkpoint_count:
        raise ValueError(
            f"Expected {expected_checkpoint_count} canary checkpoints, found {len(checkpoints)}."
        )
    payload = {
        "canary_root": str(canary_root),
        "baseline_checkpoint": str(baseline_checkpoint),
        "arms": [
            {
                "arm": arm["arm"],
                "reward_profile": arm["reward_profile"],
                "evaluation_steps": arm.get("evaluation_steps"),
                "training_seed": arm.get("seed"),
                "evaluation_seed": arm.get("evaluation_seed", manifest.get("evaluation_seed", 23)),
                "action_mode": arm.get("action_mode"),
                "action_repeat": arm.get("action_repeat"),
                "sequence_timing_profile": arm.get("sequence_timing_profile"),
            }
            for arm in manifest["arms"]
        ],
        "checkpoints": [
            {
                "arm": spec.arm,
                "reward_profile": spec.reward_profile,
                "step": spec.step,
                "path": str(spec.path),
            }
            for spec in checkpoints
        ],
        "checkpoint_count": len(checkpoints),
        "evaluation_sequence_count": len(SEQUENCES),
        "evaluation_sequences": SEQUENCES,
        "evaluation_settings": {
            "seed": 23,
            "lookahead": 1,
            "horizon_steps": 96,
            "action_mode": "direct",
            "action_repeat": 1,
            "sequence_timing_profile": "aligned",
        },
        "output_dir": str(output_dir),
    }
    return payload


def load_manifest(canary_root: Path) -> dict:
    manifest_path = canary_root / "launch_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing canary manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def discover_checkpoints(canary_root: Path) -> list[CheckpointSpec]:
    manifest = load_manifest(canary_root)
    checkpoints: list[CheckpointSpec] = []
    for arm in manifest["arms"]:
        arm_name = arm["arm"]
        reward_profile = arm["reward_profile"]
        candidates = [
            canary_root / arm_name,
            canary_root / "local_smokes" / arm_name,
        ]
        arm_root = next((path for path in candidates if path.exists()), None)
        if arm_root is None:
            raise FileNotFoundError(f"Could not find output directory for arm {arm_name!r}.")
        paths = list((arm_root / "lightweight_checkpoints").glob("**/checkpoint_*_steps.pt"))
        if not paths:
            raise FileNotFoundError(f"No lightweight checkpoints found for arm {arm_name!r} under {arm_root}.")
        seen_steps = set()
        for path in sorted(paths, key=lambda item: (checkpoint_step(item), str(item))):
            step = checkpoint_step(path)
            if step in seen_steps:
                raise ValueError(f"Duplicate lightweight checkpoint step {step} for arm {arm_name}.")
            seen_steps.add(step)
            checkpoints.append(CheckpointSpec(arm_name, reward_profile, step, path))
        missing = set(EXPECTED_CHECKPOINT_STEPS) - seen_steps
        if missing:
            raise ValueError(f"Arm {arm_name!r} is missing checkpoint steps {sorted(missing)}.")
    return sorted(checkpoints, key=lambda item: (item.step, item.arm))


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    arm_name: str,
    reward_profile: str,
    checkpoint_step: int,
    sequence_id: str,
    pitches: list[int],
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    policy = DroQPolicy.load(checkpoint, device="cpu")
    midi_path = write_sequence_midi(
        pitches,
        output_dir / "eval_midi" / f"{arm_name}_{checkpoint_step}_{sequence_id}.mid",
        midi_min=min(pitches),
        midi_max=max(pitches),
        timing=sequence_timing_from_profile("aligned"),
        fingering_fn=assign_right_hand_fingering,
        title=f"refinement canary {arm_name} {sequence_id}",
    )
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=min(pitches),
        midi_max=max(pitches),
        seed=23,
        lookahead=1,
        horizon_steps=96,
        action_mode="direct",
        action_repeat=1,
        reward_config=reward_config_from_profile(reward_profile),
    )
    obs, info = env.reset(seed=23)
    total = 0.0
    native = 0.0
    max_target = 0.0
    max_unintended = 0.0
    pressed_keys = set()
    target_vectors = []
    pressed_vectors = []
    actions = []
    fingertip_positions = []
    classifications_seen = []
    integrated_unintended = 0.0
    timesteps_above_soft = 0
    timesteps_above_press = 0
    previous_target_late_release_duration = 0
    future_target_early_activation_duration = 0
    neighbouring_wrong_press_count = 0
    unrelated_wrong_press_count = 0
    reward_components = defaultdict(list)
    completion_hits = set()
    for _ in range(env.horizon_steps):
        action, _ = policy.predict(obs, deterministic=True)
        actions.append(np.asarray(action, dtype=float))
        fingertip_positions.append(
            np.stack(list(env.fingertip_positions().values()), axis=0)
        )
        obs, reward, terminated, truncated, info = env.step(action)
        states = env.piano_key_states()
        classifications = classify_unintended_keys(
            states,
            current_target_keys=info["target_keys"],
            previous_target_keys=info.get("previous_target_keys", ()),
            future_target_keys=info.get("future_target_keys", ()),
            press_threshold=0.5,
        )
        classifications_seen.extend(classifications)
        integrated_unintended += float(info["reward_components"].get("unintended_integrated_duration", 0.0))
        timesteps_above_soft += int(float(info["max_unintended_key_state"]) > env.reward_config.unintended_soft_threshold)
        timesteps_above_press += int(float(info["max_unintended_key_state"]) >= env.reward_config.press_threshold)
        previous_target_late_release_duration += int(
            any(item.category == "previous_note_late_release" and item.is_pressed for item in classifications)
        )
        future_target_early_activation_duration += int(
            any(item.category == "future_note_early_activation" and item.is_pressed for item in classifications)
        )
        neighbouring_wrong_press_count += sum(
            1 for item in classifications if item.category == "neighbouring_key_displacement" and item.is_pressed
        )
        unrelated_wrong_press_count += sum(
            1 for item in classifications if item.category == "unrelated_key_activation" and item.is_pressed
        )
        total += float(reward)
        native += float(info["native_reward"])
        max_target = max(max_target, float(info["target_key_state"]))
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
        pressed_keys.update(info["pressed_keys"])
        for key in info["target_keys"]:
            if key in info["pressed_keys"]:
                completion_hits.add(int(key))
        target_vectors.append(binary_key_vector(info["target_keys"]))
        pressed_vectors.append(binary_key_vector(info["pressed_keys"]))
        for key, value in info["reward_components"].items():
            reward_components[key].append(float(value))
        if terminated or truncated:
            break
    target_keys = {pitch - 21 for pitch in pitches}
    pressed = pressed_key_metrics(target_keys, pressed_keys)
    timestep = timestep_key_metrics(target_vectors, pressed_vectors)
    action_metrics = action_quality(np.asarray(actions, dtype=float))
    motion_metrics = fingertip_motion_metrics(np.asarray(fingertip_positions, dtype=float))
    late_release_count = sum(1 for item in classifications_seen if item.category == "previous_note_late_release" and item.is_pressed)
    early_activation_count = sum(1 for item in classifications_seen if item.category == "future_note_early_activation" and item.is_pressed)
    wrong_press_count = len(set(pressed_keys) - target_keys)
    row = {
        "arm": arm_name,
        "reward_profile": reward_profile,
        "checkpoint_step": checkpoint_step,
        "checkpoint_path": str(checkpoint),
        "sequence_id": sequence_id,
        "sequence_pitches": json.dumps(pitches),
        "sequence_group": "single" if len(pitches) == 1 else "trained_transition",
        "pressed_keys": json.dumps(sorted(int(k) for k in pressed_keys)),
        "pressed_key_precision": pressed.precision,
        "pressed_key_recall": pressed.recall,
        "pressed_key_f1": pressed.f1,
        "timestep_precision": timestep.precision,
        "timestep_recall": timestep.recall,
        "timestep_f1": timestep.f1,
        "max_target_key_state": max_target,
        "max_unintended_key_state": max_unintended,
        "integrated_unintended_travel": integrated_unintended,
        "unintended_timesteps_above_soft_threshold": timesteps_above_soft,
        "unintended_timesteps_above_press_threshold": timesteps_above_press,
        "wrong_key_crossings": wrong_press_count,
        "previous_target_not_released_count": late_release_count,
        "previous_target_late_release_duration": previous_target_late_release_duration,
        "future_target_early_activation_count": early_activation_count,
        "future_target_early_activation_duration": future_target_early_activation_duration,
        "neighbouring_wrong_press_count": neighbouring_wrong_press_count,
        "unrelated_wrong_press_count": unrelated_wrong_press_count,
        "transition_completion": float(target_keys.issubset(pressed_keys)),
        "second_target_completion": float((pitches[-1] - 21) in completion_hits) if len(pitches) > 1 else 1.0,
        "strict_outcome": strict_outcome(target_keys, pressed_keys, max_target, max_unintended),
        "shaped_return": total,
        "native_reward_sum": native,
        **action_metrics,
        **motion_metrics,
    }
    component_rows = []
    for key, values in reward_components.items():
        arr = np.asarray(values, dtype=float)
        component_rows.append(
            {
                "arm": arm_name,
                "checkpoint_step": checkpoint_step,
                "sequence_id": sequence_id,
                "component": key,
                "mean": float(np.mean(arr)),
                "max": float(np.max(arr)),
                "total": float(np.sum(arr)),
                "nonzero_fraction": float(np.mean(np.abs(arr) > 1e-9)),
            }
        )
    return row, component_rows


def fingertip_motion_metrics(positions: np.ndarray, dt: float = 0.05) -> dict[str, float]:
    if positions.ndim != 3 or positions.shape[0] < 2:
        return {
            "mean_fingertip_speed": 0.0,
            "p95_fingertip_speed": 0.0,
            "mean_fingertip_acceleration": 0.0,
            "p95_fingertip_acceleration": 0.0,
            "mean_fingertip_jerk": 0.0,
            "p95_fingertip_jerk": 0.0,
        }
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    speed = displacement / dt
    velocity = np.diff(positions, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    acceleration_norm = np.linalg.norm(acceleration, axis=2) if acceleration.size else np.asarray([0.0])
    jerk_norm = np.linalg.norm(jerk, axis=2) if jerk.size else np.asarray([0.0])
    return {
        "mean_fingertip_speed": float(np.mean(speed)) if speed.size else 0.0,
        "p95_fingertip_speed": float(np.percentile(speed, 95)) if speed.size else 0.0,
        "mean_fingertip_acceleration": float(np.mean(acceleration_norm)),
        "p95_fingertip_acceleration": float(np.percentile(acceleration_norm, 95)),
        "mean_fingertip_jerk": float(np.mean(jerk_norm)),
        "p95_fingertip_jerk": float(np.percentile(jerk_norm, 95)),
    }


def strict_outcome(
    target_keys: set[int],
    pressed_keys: set[int],
    max_target: float,
    max_unintended: float,
) -> str:
    if target_keys.issubset(pressed_keys):
        if pressed_keys == target_keys and max_unintended < 0.25:
            return "clean_low_unintended"
        if pressed_keys == target_keys:
            return "clean_high_unintended"
        return "dirty_pressed_wrong_key"
    if max_target >= 0.25 and max_unintended < 0.25:
        return "near_clean_partial"
    return "missed"


def summarize(rows: list[dict]) -> list[dict]:
    df = defaultdict(list)
    for row in rows:
        df[(row["arm"], row["checkpoint_step"])].append(row)
    summaries = []
    for (arm, step), items in sorted(df.items()):
        all_items = items
        transition_items = [item for item in items if item["sequence_group"] == "trained_transition"]
        single_items = [item for item in items if item["sequence_group"] == "single"]
        summaries.append(
            {
                "arm": arm,
                "checkpoint_step": step,
                "overall_pressed_key_precision": float(np.mean([item["pressed_key_precision"] for item in all_items])),
                "overall_pressed_key_recall": float(np.mean([item["pressed_key_recall"] for item in all_items])),
                "overall_pressed_key_f1": float(np.mean([item["pressed_key_f1"] for item in all_items])),
                "single_anchor_f1_mean": float(np.mean([item["pressed_key_f1"] for item in single_items])),
                "single_anchor_f1_min": min(item["pressed_key_f1"] for item in single_items),
                "single_anchor_precision_mean": float(np.mean([item["pressed_key_precision"] for item in single_items])),
                "single_anchor_recall_mean": float(np.mean([item["pressed_key_recall"] for item in single_items])),
                "trained_transition_precision_mean": float(np.mean([item["pressed_key_precision"] for item in transition_items])),
                "trained_transition_recall_mean": float(np.mean([item["pressed_key_recall"] for item in transition_items])),
                "trained_transition_f1_mean": float(np.mean([item["pressed_key_f1"] for item in transition_items])),
                "trained_transition_f1_min": min(item["pressed_key_f1"] for item in transition_items),
                "worst_trained_transition": min(
                    transition_items,
                    key=lambda item: (item["pressed_key_f1"], item["timestep_f1"], -item["max_unintended_key_state"]),
                )["sequence_id"],
                "transition_completion_mean": float(np.mean([item["transition_completion"] for item in transition_items])),
                "second_target_completion_mean": float(np.mean([item["second_target_completion"] for item in transition_items])),
                "previous_target_not_released_mean": float(np.mean([item["previous_target_not_released_count"] for item in transition_items])),
                "previous_target_late_release_duration_mean": float(np.mean([item["previous_target_late_release_duration"] for item in transition_items])),
                "future_target_early_activation_duration_mean": float(np.mean([item["future_target_early_activation_duration"] for item in transition_items])),
                "wrong_key_crossings_mean": float(np.mean([item["wrong_key_crossings"] for item in all_items])),
                "neighbouring_wrong_press_count_mean": float(np.mean([item["neighbouring_wrong_press_count"] for item in all_items])),
                "unrelated_wrong_press_count_mean": float(np.mean([item["unrelated_wrong_press_count"] for item in all_items])),
                "max_unintended_key_state_mean": float(np.mean([item["max_unintended_key_state"] for item in all_items])),
                "max_unintended_key_state_max": float(np.max([item["max_unintended_key_state"] for item in all_items])),
                "integrated_unintended_travel_mean": float(np.mean([item["integrated_unintended_travel"] for item in all_items])),
                "timestep_f1_mean": float(np.mean([item["timestep_f1"] for item in all_items])),
                "timestep_f1_min": min(item["timestep_f1"] for item in all_items),
                "mean_abs_action_delta": float(np.mean([item["mean_abs_action_delta"] for item in all_items])),
                "mean_squared_action_delta": float(np.mean([item["mean_squared_action_delta"] for item in all_items])),
                "action_saturation_fraction": float(np.mean([item["action_saturation_fraction"] for item in all_items])),
                "mean_fingertip_speed": float(np.mean([item["mean_fingertip_speed"] for item in all_items])),
                "p95_fingertip_speed": float(np.mean([item["p95_fingertip_speed"] for item in all_items])),
                "mean_fingertip_acceleration": float(np.mean([item["mean_fingertip_acceleration"] for item in all_items])),
                "p95_fingertip_acceleration": float(np.mean([item["p95_fingertip_acceleration"] for item in all_items])),
                "mean_fingertip_jerk": float(np.mean([item["mean_fingertip_jerk"] for item in all_items])),
                "p95_fingertip_jerk": float(np.mean([item["p95_fingertip_jerk"] for item in all_items])),
            }
        )
    return summaries


def compare_to_control(summary: list[dict]) -> list[dict]:
    by_key = {(row["arm"], row["checkpoint_step"]): row for row in summary}
    rows = []
    for row in summary:
        if row["arm"] in {"control_continue_sensitive_v1", "frozen_800k_baseline"}:
            continue
        control = by_key.get(("control_continue_sensitive_v1", row["checkpoint_step"]))
        if control is None:
            continue
        diff = {"arm": row["arm"], "checkpoint_step": row["checkpoint_step"], "control_arm": control["arm"]}
        for key, value in row.items():
            if isinstance(value, (int, float)) and key != "checkpoint_step":
                diff[f"{key}_delta_vs_control"] = float(value) - float(control[key])
        rows.append(diff)
    return rows


def compare_arms(summary: list[dict], treatment_arm: str, control_arm: str, _filename: str) -> list[dict]:
    by_key = {(row["arm"], row["checkpoint_step"]): row for row in summary}
    rows = []
    for row in summary:
        if row["arm"] != treatment_arm:
            continue
        control = by_key.get((control_arm, row["checkpoint_step"]))
        if control is None:
            continue
        diff = {
            "treatment_arm": treatment_arm,
            "control_arm": control_arm,
            "checkpoint_step": row["checkpoint_step"],
        }
        for key, value in row.items():
            if isinstance(value, (int, float)) and key != "checkpoint_step":
                diff[f"{key}_treatment"] = float(value)
                diff[f"{key}_control"] = float(control[key])
                diff[f"{key}_delta"] = float(value) - float(control[key])
        rows.append(diff)
    return rows


def compare_to_baseline(summary: list[dict]) -> list[dict]:
    baseline = next(
        (row for row in summary if row["arm"] == "frozen_800k_baseline" and row["checkpoint_step"] == 0),
        None,
    )
    if baseline is None:
        return []
    rows = []
    for row in summary:
        if row["arm"] == "frozen_800k_baseline":
            continue
        diff = {
            "arm": row["arm"],
            "checkpoint_step": row["checkpoint_step"],
            "baseline_arm": baseline["arm"],
            "baseline_step": baseline["checkpoint_step"],
        }
        for key, value in row.items():
            if isinstance(value, (int, float)) and key != "checkpoint_step":
                diff[f"{key}_candidate"] = float(value)
                diff[f"{key}_baseline"] = float(baseline[key])
                diff[f"{key}_delta_vs_step0"] = float(value) - float(baseline[key])
        rows.append(diff)
    return rows


def select_candidate(summary: list[dict]) -> dict:
    baseline = next(
        row for row in summary if row["arm"] == "frozen_800k_baseline" and row["checkpoint_step"] == 0
    )
    candidates = [row for row in summary if row["arm"] != "frozen_800k_baseline"]
    annotated = []
    for row in candidates:
        rejection_reasons = []
        if row["single_anchor_f1_min"] < 0.99:
            rejection_reasons.append("major_anchor_regression")
        if row["overall_pressed_key_precision"] < baseline["overall_pressed_key_precision"] - 0.10:
            rejection_reasons.append("severe_precision_regression")
        if row["wrong_key_crossings_mean"] > baseline["wrong_key_crossings_mean"] + 1.0:
            rejection_reasons.append("severe_wrong_key_regression")
        if row["trained_transition_f1_min"] < baseline["trained_transition_f1_min"] - 0.20:
            rejection_reasons.append("severe_worst_transition_regression")
        if row["timestep_f1_mean"] < baseline["timestep_f1_mean"] - 0.10:
            rejection_reasons.append("severe_timing_regression")
        annotated.append(
            {
                "arm": row["arm"],
                "checkpoint_step": row["checkpoint_step"],
                "viable": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
                "metrics": {
                    key: row[key]
                    for key in (
                        "overall_pressed_key_precision",
                        "overall_pressed_key_recall",
                        "overall_pressed_key_f1",
                        "single_anchor_f1_min",
                        "trained_transition_f1_mean",
                        "trained_transition_f1_min",
                        "transition_completion_mean",
                        "second_target_completion_mean",
                        "previous_target_not_released_mean",
                        "wrong_key_crossings_mean",
                        "max_unintended_key_state_mean",
                        "integrated_unintended_travel_mean",
                        "timestep_f1_mean",
                        "mean_abs_action_delta",
                        "action_saturation_fraction",
                        "p95_fingertip_jerk",
                    )
                },
            }
        )
    viable = [item for item in annotated if item["viable"]]
    selected = None
    if viable:
        selected = max(
            viable,
            key=lambda item: (
                item["metrics"]["transition_completion_mean"],
                item["metrics"]["second_target_completion_mean"],
                -item["metrics"]["previous_target_not_released_mean"],
                item["metrics"]["trained_transition_f1_mean"],
                -item["metrics"]["wrong_key_crossings_mean"],
                -item["metrics"]["max_unintended_key_state_mean"],
                -item["metrics"]["mean_abs_action_delta"],
            ),
        )
    beats_baseline = False
    if selected is not None:
        metrics = selected["metrics"]
        beats_baseline = (
            metrics["transition_completion_mean"] > baseline["transition_completion_mean"]
            and metrics["single_anchor_f1_min"] >= baseline["single_anchor_f1_min"]
            and metrics["overall_pressed_key_precision"] >= baseline["overall_pressed_key_precision"] - 0.05
            and metrics["wrong_key_crossings_mean"] <= baseline["wrong_key_crossings_mean"] + 0.25
            and metrics["timestep_f1_mean"] >= baseline["timestep_f1_mean"] - 0.10
        )
    return {
        "selection_policy": [
            "Reject major anchor regression: single_anchor_f1_min < 0.99.",
            "Reject severe precision regression: overall precision more than 0.10 below step-0.",
            "Reject severe wrong-key regression: wrong-key crossings mean more than 1.0 above step-0.",
            "Reject severe worst-transition regression: trained_transition_f1_min more than 0.20 below step-0.",
            "Reject severe timing regression: timestep_f1_mean more than 0.10 below step-0.",
            "Among viable candidates prioritize transition completion, second-target completion, lower late release, transition F1, fewer wrong keys, lower unintended travel, then smoother actions.",
        ],
        "baseline": {
            "arm": baseline["arm"],
            "checkpoint_step": baseline["checkpoint_step"],
            "metrics": {
                key: baseline[key]
                for key in (
                    "overall_pressed_key_precision",
                    "overall_pressed_key_recall",
                    "overall_pressed_key_f1",
                    "single_anchor_f1_min",
                    "trained_transition_f1_mean",
                    "trained_transition_f1_min",
                    "transition_completion_mean",
                    "second_target_completion_mean",
                    "previous_target_not_released_mean",
                    "wrong_key_crossings_mean",
                    "max_unintended_key_state_mean",
                    "integrated_unintended_travel_mean",
                    "timestep_f1_mean",
                    "mean_abs_action_delta",
                    "action_saturation_fraction",
                    "p95_fingertip_jerk",
                )
            },
        },
        "candidates": annotated,
        "selected_candidate": selected,
        "does_selected_clearly_beat_step0": beats_baseline,
        "decision": "replace_baseline" if beats_baseline else "retain_frozen_baseline",
    }


def checkpoint_step(path: Path) -> int:
    match = re.search(r"(?:full_)?checkpoint_(\d+)_steps\.pt$", path.name)
    if not match:
        return 0
    return int(match.group(1))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    summary: list[dict],
    comparison: list[dict],
    ab_delta: list[dict],
    bc_delta: list[dict],
    baseline_delta: list[dict],
    recommendation: dict,
) -> None:
    best = recommendation.get("selected_candidate")
    decision = recommendation.get("decision")
    lines = [
        "# Refinement Canary Comparison",
        "",
        "This report evaluates the three-arm warm-start refinement canary using common behavioural metrics. It is not a clean from-scratch causal algorithm/reward study.",
        "",
        "## Selection Decision",
        "",
        f"- decision: `{decision}`",
        f"- selected candidate: `{best['arm']} @ {best['checkpoint_step']}`" if best else "- selected candidate: `none`",
        f"- clearly beats step-0 baseline: `{recommendation.get('does_selected_clearly_beat_step0')}`",
        "",
        "The explicit selection policy rejects major anchor, precision, wrong-key, worst-transition and timing regressions before comparing viable candidates on transition completion, release behaviour, F1, unintended-key travel and motion smoothness.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## A vs B: Release/Completion Shaping",
        "",
        delta_snapshot(
            ab_delta,
            [
                "transition_completion_mean_delta",
                "second_target_completion_mean_delta",
                "previous_target_not_released_mean_delta",
                "trained_transition_f1_mean_delta",
                "wrong_key_crossings_mean_delta",
                "max_unintended_key_state_mean_delta",
            ],
        ),
        "",
        "## B vs C: Motion Penalties",
        "",
        delta_snapshot(
            bc_delta,
            [
                "mean_abs_action_delta_delta",
                "action_saturation_fraction_delta",
                "p95_fingertip_jerk_delta",
                "transition_completion_mean_delta",
                "trained_transition_f1_mean_delta",
                "timestep_f1_mean_delta",
            ],
        ),
        "",
        "## Treatment Deltas Versus Matched Control",
        "",
        markdown_table(comparison) if comparison else "No matched treatment/control checkpoints were available.",
        "",
        "## Deltas Versus Step-0 Frozen Baseline",
        "",
        delta_snapshot(
            baseline_delta,
            [
                "overall_pressed_key_precision_delta_vs_step0",
                "overall_pressed_key_recall_delta_vs_step0",
                "overall_pressed_key_f1_delta_vs_step0",
                "transition_completion_mean_delta_vs_step0",
                "second_target_completion_mean_delta_vs_step0",
                "previous_target_not_released_mean_delta_vs_step0",
                "wrong_key_crossings_mean_delta_vs_step0",
                "max_unintended_key_state_mean_delta_vs_step0",
            ],
        ),
        "",
        "## Answers",
        "",
        report_answers(summary, ab_delta, bc_delta, recommendation),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def delta_snapshot(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return "No rows available."
    compact = []
    for row in rows:
        compact.append(
            {
                key: row.get(key)
                for key in ("arm", "treatment_arm", "control_arm", "checkpoint_step", *fields)
                if key in row
            }
        )
    return markdown_table(compact)


def report_answers(
    summary: list[dict],
    ab_delta: list[dict],
    bc_delta: list[dict],
    recommendation: dict,
) -> str:
    def mean_delta(rows: list[dict], key: str) -> float:
        values = [float(row[key]) for row in rows if key in row]
        return float(np.mean(values)) if values else 0.0

    b_completion = mean_delta(ab_delta, "transition_completion_mean_delta")
    b_second = mean_delta(ab_delta, "second_target_completion_mean_delta")
    b_anchor_delta = mean_delta(ab_delta, "single_anchor_f1_min_delta")
    c_action_delta = mean_delta(bc_delta, "mean_abs_action_delta_delta")
    c_saturation_delta = mean_delta(bc_delta, "action_saturation_fraction_delta")
    c_completion_delta = mean_delta(bc_delta, "transition_completion_mean_delta")
    selected = recommendation.get("selected_candidate")
    selected_text = f"{selected['arm']} @ {selected['checkpoint_step']}" if selected else "none"
    lines = [
        f"- Did B outperform A on release/completion? Mean transition completion delta `{b_completion:.3f}`, second-target completion delta `{b_second:.3f}`.",
        f"- Did B preserve anchors? Mean single-anchor-min F1 delta `{b_anchor_delta:.3f}`.",
        f"- Did C improve motion quality relative to B? Mean action-delta delta `{c_action_delta:.3f}`, saturation delta `{c_saturation_delta:.3f}`; negative is smoother/lower saturation.",
        f"- Did C trade away note completion? Mean transition-completion delta versus B `{c_completion_delta:.3f}`.",
        f"- Balanced candidate by policy: `{selected_text}`.",
        f"- Final decision: `{recommendation.get('decision')}`.",
    ]
    return "\n".join(lines)


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
