#!/usr/bin/env python
"""Evaluate five-note factorial checkpoints for one completed run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.evaluation import binary_key_vector, pressed_key_metrics, timestep_key_metrics
from ala_pianist.evaluation.unintended import classify_unintended_keys
from ala_pianist.experiments.five_note_factorial import FIVE_NOTE_EVALUATION_SEQUENCES
from ala_pianist.music import assign_right_hand_fingering, sequence_timing_from_profile, write_sequence_midi
from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv
from evaluate_general_one_hand_policy import reward_config_from_profile


ROOT = Path("/home/reece_dev/msc-audio-pianist")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_step(path: Path) -> int:
    match = re.search(r"(?:checkpoint|full_checkpoint)_(\d+)_steps", path.name)
    if not match:
        return -1
    return int(match.group(1))


def discover_checkpoints(run_dir: Path, algorithm: str) -> list[Path]:
    suffixes = ("*.pt",) if algorithm == "droq" else ("*.zip",)
    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.extend(run_dir.glob(f"**/{suffix}"))
    checkpoints = [
        path
        for path in candidates
        if checkpoint_step(path) > 0
        and ("lightweight" in str(path) or "checkpoint_" in path.name)
        and "replay_buffer" not in path.name
        and "full_checkpoint" not in path.name
        and "rolling_latest" not in path.name
    ]
    return sorted(checkpoints, key=checkpoint_step)


def load_policy(path: Path, algorithm: str):
    if algorithm == "droq":
        return DroQPolicy.load(path, device="cpu")
    if algorithm == "sac":
        return SAC.load(path)
    raise ValueError(f"Unsupported algorithm {algorithm!r}.")


def write_eval_midi(out_dir: Path, name: str, pitches: tuple[int, ...]) -> Path:
    return write_sequence_midi(
        list(pitches),
        out_dir / "eval_midi" / f"{name}_{'_'.join(map(str, pitches))}.mid",
        midi_min=min(pitches),
        midi_max=max(pitches),
        timing=sequence_timing_from_profile("aligned"),
        fingering_fn=assign_right_hand_fingering,
        title=f"five note factorial {name}",
    )


def sequence_group(name: str) -> str:
    if name.startswith("single_"):
        return "trained_single"
    if name.startswith("trained_"):
        return "trained_transition"
    if name.startswith("heldout_"):
        return "heldout_transition"
    if name.startswith("composition_"):
        return "composition_probe"
    if name.startswith("repeat_"):
        return "repeated_note_probe"
    return "unknown"


def strict_outcome(target_keys: set[int], pressed_keys: set[int], max_target: float, max_unintended: float) -> str:
    if target_keys and target_keys.issubset(pressed_keys) and not (pressed_keys - target_keys):
        return "clean_low_unintended" if max_unintended < 0.25 else "clean_high_unintended"
    if target_keys and target_keys.intersection(pressed_keys):
        return "dirty_pressed_wrong_key"
    if max_target >= 0.25 and max_unintended < 0.25:
        return "near_clean_partial"
    return "missed"


def evaluate_checkpoint(path: Path, config: dict[str, Any], output_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    policy = load_policy(path, config["algorithm"])
    reward_config = reward_config_from_profile(config["reward_profile"])
    step = checkpoint_step(path)
    rows: list[dict] = []
    reward_rows: list[dict] = []
    health_rows: list[dict] = []
    for name, pitches in FIVE_NOTE_EVALUATION_SEQUENCES.items():
        midi_path = write_eval_midi(output_dir, name, pitches)
        env = GeneralOneHandGoalEnv(
            midi_path=midi_path,
            midi_min=72,
            midi_max=76,
            seed=config["seed"],
            lookahead=config["lookahead"],
            horizon_steps=128,
            action_mode=config["action_mode"],
            action_repeat=config["action_repeat"],
            reward_config=reward_config,
        )
        obs, info = env.reset(seed=config["seed"])
        total = 0.0
        native = 0.0
        max_target = 0.0
        max_unintended = 0.0
        pressed_keys: set[int] = set()
        target_vectors = []
        pressed_vectors = []
        classifications_seen = []
        reward_components = defaultdict(list)
        actions = []
        for _ in range(env.horizon_steps):
            action, _ = policy.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            states = env.piano_key_states()
            classifications = classify_unintended_keys(
                states,
                current_target_keys=info["target_keys"],
                previous_target_keys=info.get("previous_target_keys", ()),
                future_target_keys=info.get("future_target_keys", ()),
                press_threshold=reward_config.press_threshold,
            )
            classifications_seen.extend(classifications)
            total += float(reward)
            native += float(info["native_reward"])
            max_target = max(max_target, float(info["target_key_state"]))
            max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
            pressed_keys.update(info["pressed_keys"])
            target_vectors.append(binary_key_vector(info["target_keys"]))
            pressed_vectors.append(binary_key_vector(info["pressed_keys"]))
            for key, value in info["reward_components"].items():
                reward_components[key].append(float(value))
            if terminated or truncated:
                break
        target_keys = {pitch - 21 for pitch in set(pitches)}
        pressed_metrics = pressed_key_metrics(target_keys, pressed_keys)
        timestep_metrics = timestep_key_metrics(target_vectors, pressed_vectors)
        unintended_values = [item.value for item in classifications_seen if item.value >= reward_config.unintended_soft_threshold]
        wrong_pressed = pressed_keys - target_keys
        category_counts = defaultdict(int)
        category_duration = defaultdict(float)
        worst_key = None
        worst_value = -1.0
        for item in classifications_seen:
            excess = max(0.0, item.value - reward_config.unintended_soft_threshold)
            if item.value >= reward_config.unintended_soft_threshold:
                category_counts[item.category] += 1
                category_duration[item.category] += excess
            if item.value > worst_value:
                worst_key = item.key_index
                worst_value = item.value
        rows.append(
            {
                "checkpoint_step": step,
                "checkpoint_path": str(path),
                "sequence_name": name,
                "sequence_group": sequence_group(name),
                "sequence_pitches": json.dumps(list(pitches)),
                "expected_pressed_key_set": json.dumps(sorted(target_keys)),
                "observed_pressed_key_set": json.dumps(sorted(pressed_keys)),
                "pressed_key_precision": pressed_metrics.precision,
                "pressed_key_recall": pressed_metrics.recall,
                "pressed_key_f1": pressed_metrics.f1,
                "timestep_precision": timestep_metrics.precision,
                "timestep_recall": timestep_metrics.recall,
                "timestep_f1": timestep_metrics.f1,
                "strict_outcome": strict_outcome(target_keys, pressed_keys, max_target, max_unintended),
                "incorrect_threshold_crossing_count": len(wrong_pressed),
                "missed_target_count": len(target_keys - pressed_keys),
                "max_unintended_key_state": max_unintended,
                "integrated_unintended_travel": float(sum(unintended_values)),
                "timesteps_above_unintended_soft_threshold": len(unintended_values),
                "timesteps_above_press_threshold": sum(1 for item in classifications_seen if item.is_pressed),
                "neighbouring_key_event_count": category_counts["neighbouring_key_displacement"],
                "unrelated_key_event_count": category_counts["unrelated_key_activation"],
                "previous_target_late_release_count": category_counts["previous_note_late_release"],
                "previous_target_late_release_duration": category_duration["previous_note_late_release"],
                "future_target_early_activation_count": category_counts["future_note_early_activation"],
                "future_target_early_activation_duration": category_duration["future_note_early_activation"],
                "mean_event_duration": float(np.mean(unintended_values)) if unintended_values else 0.0,
                "maximum_event_duration": float(np.max(unintended_values)) if unintended_values else 0.0,
                "normalised_event_rate_per_1000_timesteps": 1000.0 * len(unintended_values) / max(1, len(target_vectors)),
                "episode_affected_percentage": 100.0 if unintended_values else 0.0,
                "worst_unintended_key": worst_key,
                "shaped_return": total,
                "native_reward_sum": native,
                "deterministic_action_saturation": float(np.mean(np.abs(np.asarray(actions)) >= 0.95)) if actions else 0.0,
            }
        )
        for component, values in reward_components.items():
            arr = np.asarray(values, dtype=float)
            reward_rows.append(
                {
                    "checkpoint_step": step,
                    "sequence_name": name,
                    "component": component,
                    "mean": float(np.mean(arr)),
                    "max": float(np.max(arr)),
                    "total": float(np.sum(arr)),
                }
            )
    health_rows.append(
        {
            "checkpoint_step": step,
            "checkpoint_path": str(path),
            "checkpoint_size_bytes": path.stat().st_size,
            "algorithm": config["algorithm"],
            "reward_profile": config["reward_profile"],
            "device": "evaluation_cpu",
        }
    )
    return rows, reward_rows, health_rows


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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["checkpoint_step"], row["sequence_group"])].append(row)
    summaries = []
    for (step, group), items in grouped.items():
        summaries.append(
            {
                "checkpoint_step": step,
                "sequence_group": group,
                "mean_pressed_key_f1": float(np.mean([row["pressed_key_f1"] for row in items])),
                "mean_timestep_f1": float(np.mean([row["timestep_f1"] for row in items])),
                "mean_integrated_unintended_travel": float(np.mean([row["integrated_unintended_travel"] for row in items])),
                "mean_max_unintended_key_state": float(np.mean([row["max_unintended_key_state"] for row in items])),
                "mean_wrong_key_crossings": float(np.mean([row["incorrect_threshold_crossing_count"] for row in items])),
                "affected_episode_percentage": float(np.mean([row["episode_affected_percentage"] for row in items])),
            }
        )
    return sorted(summaries, key=lambda row: (row["checkpoint_step"], row["sequence_group"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", action="append", default=[])
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = load_json(Path(args.config))
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [Path(path) for path in args.checkpoint] or discover_checkpoints(run_dir, config["algorithm"])
    if not checkpoints:
        raise SystemExit(f"No {config['algorithm']} lightweight checkpoints found under {run_dir}.")
    all_rows: list[dict] = []
    all_reward_rows: list[dict] = []
    all_health_rows: list[dict] = []
    for checkpoint in checkpoints:
        rows, reward_rows, health_rows = evaluate_checkpoint(checkpoint, config, output_dir)
        all_rows.extend(rows)
        all_reward_rows.extend(reward_rows)
        all_health_rows.extend(health_rows)
    summary_rows = summarize(all_rows)
    write_csv(output_dir / "per_checkpoint_per_sequence_metrics.csv", all_rows)
    write_csv(output_dir / "per_checkpoint_summary.csv", summary_rows)
    write_csv(output_dir / "per_checkpoint_training_health.csv", all_health_rows)
    write_csv(output_dir / "reward_component_summary.csv", all_reward_rows)
    best = max(summary_rows, key=lambda row: (row["mean_pressed_key_f1"], row["mean_timestep_f1"]))
    payload = {
        "run_dir": str(run_dir),
        "config": str(args.config),
        "checkpoints": [str(path) for path in checkpoints],
        "best_checkpoint_by_metric": best,
        "deterministic": True,
    }
    (output_dir / "evaluation_config.json").write_text(json.dumps({"config": config}, indent=2), encoding="utf-8")
    (output_dir / "best_checkpoint_by_metric.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "evaluation_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "checkpoint_count": len(checkpoints)}, indent=2))


if __name__ == "__main__":
    main()
