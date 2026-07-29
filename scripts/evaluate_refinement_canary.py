#!/usr/bin/env python
"""Evaluate refinement-canary arms with behavior and motion-quality metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-root", type=Path, default=CANARY)
    parser.add_argument("--output-dir", type=Path, default=CANARY / "validation")
    parser.add_argument("--include-frozen-baseline", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.canary_root, args.output_dir, include_frozen=args.include_frozen_baseline)
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate(canary_root: Path, output_dir: Path, *, include_frozen: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((canary_root / "launch_manifest.json").read_text(encoding="utf-8"))
    checkpoints = []
    if include_frozen:
        checkpoints.append(("frozen_800k_baseline", "transition_cleanup_sensitive_v1", 800000, FROZEN_CKPT))
    for arm in manifest["arms"]:
        arm_name = arm["arm"]
        reward_profile = arm["reward_profile"]
        arm_root = canary_root / "local_smokes" / arm_name
        paths = list((arm_root / "lightweight_checkpoints").glob("**/checkpoint_*_steps.pt"))
        paths.extend((arm_root / "checkpoints").glob("**/checkpoint_*_steps.pt"))
        paths.extend((arm_root / "checkpoints").glob("**/full_checkpoint_*_steps.pt"))
        seen = set()
        for path in sorted(paths, key=lambda item: (checkpoint_step(item), str(item))):
            key = (checkpoint_step(path), path.name)
            if key in seen:
                continue
            seen.add(key)
            checkpoints.append((arm_name, reward_profile, checkpoint_step(path), path))
    rows = []
    reward_rows = []
    for arm_name, reward_profile, step, checkpoint in checkpoints:
        for sequence_id, pitches in SEQUENCES.items():
            row, components = evaluate_checkpoint(
                checkpoint,
                arm_name=arm_name,
                reward_profile=reward_profile,
                checkpoint_step=step,
                sequence_id=sequence_id,
                pitches=pitches,
                output_dir=output_dir,
            )
            rows.append(row)
            reward_rows.extend(components)
    summary = summarize(rows)
    comparison = compare_to_control(summary)
    write_csv(output_dir / "per_checkpoint_per_sequence_metrics.csv", rows)
    write_csv(output_dir / "reward_component_summary.csv", reward_rows)
    write_csv(output_dir / "per_checkpoint_summary.csv", summary)
    write_csv(output_dir / "arm_comparison.csv", comparison)
    report_path = output_dir / "comparison_report.md"
    write_report(report_path, summary, comparison)
    payload = {
        "evaluated_checkpoints": [str(item[3]) for item in checkpoints],
        "outputs": {
            "per_checkpoint_per_sequence_metrics": str(output_dir / "per_checkpoint_per_sequence_metrics.csv"),
            "reward_component_summary": str(output_dir / "reward_component_summary.csv"),
            "per_checkpoint_summary": str(output_dir / "per_checkpoint_summary.csv"),
            "arm_comparison": str(output_dir / "arm_comparison.csv"),
            "comparison_report": str(report_path),
        },
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


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
        "wrong_key_crossings": wrong_press_count,
        "previous_target_not_released_count": late_release_count,
        "transition_completion": float(target_keys.issubset(pressed_keys)),
        "second_target_completion": float((pitches[-1] - 21) in completion_hits) if len(pitches) > 1 else 1.0,
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
        return {"p95_fingertip_jerk": 0.0}
    velocity = np.diff(positions, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    if jerk.size == 0:
        return {"p95_fingertip_jerk": 0.0}
    return {"p95_fingertip_jerk": float(np.percentile(np.linalg.norm(jerk, axis=2), 95))}


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
                "single_anchor_f1_min": min(item["pressed_key_f1"] for item in single_items),
                "trained_transition_f1_mean": float(np.mean([item["pressed_key_f1"] for item in transition_items])),
                "trained_transition_f1_min": min(item["pressed_key_f1"] for item in transition_items),
                "transition_completion_mean": float(np.mean([item["transition_completion"] for item in transition_items])),
                "second_target_completion_mean": float(np.mean([item["second_target_completion"] for item in transition_items])),
                "previous_target_not_released_mean": float(np.mean([item["previous_target_not_released_count"] for item in transition_items])),
                "wrong_key_crossings_mean": float(np.mean([item["wrong_key_crossings"] for item in all_items])),
                "timestep_f1_mean": float(np.mean([item["timestep_f1"] for item in all_items])),
                "mean_abs_action_delta": float(np.mean([item["mean_abs_action_delta"] for item in all_items])),
                "action_saturation_fraction": float(np.mean([item["action_saturation_fraction"] for item in all_items])),
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


def write_report(path: Path, summary: list[dict], comparison: list[dict]) -> None:
    lines = [
        "# Refinement Canary Comparison",
        "",
        "This report compares available canary checkpoints. It is valid for local smoke outputs or the later 75k canary outputs.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Treatment Deltas Versus Matched Control",
        "",
        markdown_table(comparison) if comparison else "No matched treatment/control checkpoints were available.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
