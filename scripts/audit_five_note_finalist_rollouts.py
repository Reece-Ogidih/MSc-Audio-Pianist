#!/usr/bin/env python
"""Visual and repeated-rollout audit for the five-note finalist controllers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from stable_baselines3 import SAC

ROOT = Path("/home/reece_dev/msc-audio-pianist")
sys.path.insert(0, str(ROOT / "scripts"))

from ala_pianist.evaluation import binary_key_vector, pressed_key_metrics, timestep_key_metrics  # noqa: E402
from ala_pianist.evaluation.unintended import classify_unintended_keys  # noqa: E402
from ala_pianist.experiments.five_note_factorial import FIVE_NOTE_EVALUATION_SEQUENCES  # noqa: E402
from ala_pianist.music import assign_right_hand_fingering, sequence_timing_from_profile, write_sequence_midi  # noqa: E402
from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv  # noqa: E402
from evaluate_five_note_factorial_checkpoint_sweep import sequence_group, strict_outcome  # noqa: E402
from evaluate_general_one_hand_policy import reward_config_from_profile  # noqa: E402
from render_frozen_five_note_smoke_rollout import control_timestep, trace_row, write_plot, write_trace  # noqa: E402


FROZEN_RELEASE = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1"
EXPORT_ROOT = ROOT / "artifacts/five_note_factorial_1m_hex_export/extracted"
ANALYSIS_DIR = ROOT / "artifacts/five_note_factorial_1m_hex_export/analysis"
AUDIT_DIR = FROZEN_RELEASE / "audit"
PRESS_THRESHOLD = 0.5
SOFT_THRESHOLD = 0.2
ACTION_SATURATION_THRESHOLD = 0.95
ACTION_SATURATION_FRACTION_THRESHOLD = 0.05
TARGET_BRIEF_PRESS_STEPS = 2


@dataclass(frozen=True)
class Finalist:
    model_id: str
    condition_id: str
    algorithm: str
    reward_profile: str
    checkpoint_step: int
    checkpoint_path: Path
    config_path: Path


class PolicyAdapter:
    def __init__(self, model: Any, algorithm: str):
        self.model = model
        self.algorithm = algorithm

    def predict(self, observation, *, deterministic: bool):
        if self.algorithm == "droq":
            return self.model.predict(observation, deterministic=deterministic)
        return self.model.predict(observation, deterministic=deterministic)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def note_name(midi: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[int(midi) % 12]}{int(midi) // 12 - 1}"


def key_to_midi(key_index: int) -> int:
    return int(key_index) + 21


def audit_sequences() -> dict[str, tuple[int, ...]]:
    return {
        name: pitches
        for name, pitches in FIVE_NOTE_EVALUATION_SEQUENCES.items()
        if sequence_group(name)
        in {"trained_single", "trained_transition", "heldout_transition", "composition_probe"}
    }


def deterministic_seed_list(count: int = 20, *, base_seed: int = 13) -> list[int]:
    if count < 1:
        raise ValueError("count must be positive")
    return [int(base_seed + idx) for idx in range(count)]


def finalist_definitions() -> list[Finalist]:
    manifest = load_json(FROZEN_RELEASE / "manifest.json")
    comparison = pd.read_csv(ANALYSIS_DIR / "per_condition_comparison.csv").set_index("condition_id")
    source_map = {
        "droq_sensitive_v1": {
            "model_id": "droq_sensitive_v1_800k_frozen",
            "algorithm": "droq",
            "reward_profile": "transition_cleanup_sensitive_v1",
            "step": 800000,
            "checkpoint": FROZEN_RELEASE / manifest["frozen_checkpoint_relative_path"],
            "config": FROZEN_RELEASE / "provenance/resolved_training_config.json",
        },
        "sac_sensitive_v1": {
            "model_id": "sac_sensitive_v1_700k",
            "algorithm": "sac",
            "reward_profile": "transition_cleanup_sensitive_v1",
            "step": 700000,
            "checkpoint": Path(comparison.loc["sac_sensitive_v1", "selected_local_checkpoint_path"]),
            "config": EXPORT_ROOT
            / "garlick/sac_sensitive_v1/sac_sensitive_v1_five_note_seed13_1m_20260723T191446Z/resolved_training_config.json",
        },
        "sac_original": {
            "model_id": "sac_original_900k",
            "algorithm": "sac",
            "reward_profile": "transition_cleanup",
            "step": 900000,
            "checkpoint": Path(comparison.loc["sac_original", "selected_local_checkpoint_path"]),
            "config": EXPORT_ROOT
            / "aching/sac_original/sac_original_five_note_seed13_1m_20260723T205625Z/resolved_training_config.json",
        },
    }
    finalists = []
    for condition_id, info in source_map.items():
        checkpoint = Path(info["checkpoint"])
        config = Path(info["config"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if not config.exists():
            raise FileNotFoundError(config)
        finalists.append(
            Finalist(
                model_id=str(info["model_id"]),
                condition_id=condition_id,
                algorithm=str(info["algorithm"]),
                reward_profile=str(info["reward_profile"]),
                checkpoint_step=int(info["step"]),
                checkpoint_path=checkpoint,
                config_path=config,
            )
        )
    return finalists


def load_policy(finalist: Finalist) -> PolicyAdapter:
    if finalist.algorithm == "droq":
        return PolicyAdapter(DroQPolicy.load(finalist.checkpoint_path, device="cpu"), "droq")
    if finalist.algorithm == "sac":
        return PolicyAdapter(SAC.load(finalist.checkpoint_path, device="cpu"), "sac")
    raise ValueError(f"Unknown finalist algorithm {finalist.algorithm!r}.")


def write_eval_midi(path: Path, sequence_id: str, pitches: tuple[int, ...], timing_profile: str) -> Path:
    return write_sequence_midi(
        list(pitches),
        path / f"{sequence_id}_{'_'.join(str(p) for p in pitches)}.mid",
        midi_min=min(pitches),
        midi_max=max(pitches),
        timing=sequence_timing_from_profile(timing_profile),
        fingering_fn=assign_right_hand_fingering,
        title=f"five note finalist audit {sequence_id}",
    )


def build_env(
    finalist: Finalist,
    *,
    sequence_id: str,
    pitches: tuple[int, ...],
    seed: int,
    midi_dir: Path,
) -> GeneralOneHandGoalEnv:
    config = load_json(finalist.config_path)
    midi_path = write_eval_midi(midi_dir, sequence_id, pitches, config["sequence_timing_profile"])
    return GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=min(pitches),
        midi_max=max(pitches),
        seed=seed,
        lookahead=config["lookahead"],
        horizon_steps=128,
        action_mode=config["action_mode"],
        action_repeat=config["action_repeat"],
        reward_config=reward_config_from_profile(finalist.reward_profile),
    )


def failure_labels_from_trace(
    *,
    pitches: tuple[int, ...],
    rows: list[dict[str, Any]],
    pressed_keys: set[int],
    timestep_f1: float,
    action_saturation: float,
) -> list[str]:
    labels: set[str] = set()
    target_keys = [pitch - 21 for pitch in pitches]
    unique_target_keys = set(target_keys)
    if not unique_target_keys.intersection(pressed_keys):
        labels.add("target_never_pressed")
    if target_keys and target_keys[0] not in pressed_keys:
        labels.add("first_target_missed")
    if len(target_keys) >= 2 and target_keys[1] not in pressed_keys:
        labels.add("second_target_missed")
    if len(unique_target_keys) > 1 and not unique_target_keys.issubset(pressed_keys):
        labels.add("transition_incomplete")
    wrong_keys = pressed_keys - unique_target_keys
    if wrong_keys:
        neighbour_keys = {target + delta for target in unique_target_keys for delta in (-1, 1)}
        if wrong_keys & neighbour_keys:
            labels.add("neighbouring_wrong_key_press")
        if wrong_keys - neighbour_keys:
            labels.add("unrelated_wrong_key_press")
        if len(wrong_keys) >= 2:
            labels.add("multiple_unintended_keys")
    active_target_by_key = Counter()
    pressed_by_key = Counter()
    late_release = False
    early_activation = False
    for row in rows:
        active_targets = set(json.loads(row["intended_key_indices"]))
        row_pressed = set(json.loads(row["pressed_key_indices"]))
        for key in active_targets:
            active_target_by_key[int(key)] += 1
        for key in row_pressed:
            pressed_by_key[int(key)] += 1
        events = json.loads(row["wrong_key_threshold_crossing_events"])
        if any(event["category"] == "previous_note_late_release" for event in events):
            late_release = True
        if any(event["category"] == "future_note_early_activation" for event in events):
            early_activation = True
    if early_activation:
        labels.add("early_target_activation")
    if late_release:
        labels.add("previous_target_not_released")
    for key in unique_target_keys:
        if pressed_by_key[key] and pressed_by_key[key] < TARGET_BRIEF_PRESS_STEPS:
            labels.add("target_press_too_brief")
    if timestep_f1 < 0.5 and unique_target_keys.issubset(pressed_keys):
        labels.add("late_target_activation")
    if action_saturation >= ACTION_SATURATION_FRACTION_THRESHOLD:
        labels.add("action_saturation")
    if not labels:
        labels.add("none")
    return sorted(labels)


def rollout_once(
    *,
    finalist: Finalist,
    policy: PolicyAdapter,
    sequence_id: str,
    pitches: tuple[int, ...],
    seed: int,
    deterministic: bool,
    midi_dir: Path,
    render_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    env = build_env(finalist, sequence_id=sequence_id, pitches=pitches, seed=seed, midi_dir=midi_dir)
    dt = control_timestep(env)
    obs, info = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    target_vectors: list[np.ndarray] = []
    pressed_vectors: list[np.ndarray] = []
    pressed_keys: set[int] = set()
    classifications_seen = []
    reward_components = defaultdict(list)
    actions = []
    total_reward = 0.0
    native_reward = 0.0
    max_target = 0.0
    max_unintended = 0.0
    terminated = truncated = False
    step = 0
    while not (terminated or truncated) and step < env.horizon_steps:
        action, _ = policy.predict(obs, deterministic=deterministic)
        action = np.asarray(action, dtype=np.float32)
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        native_reward += float(info["native_reward"])
        states = env.piano_key_states()
        classifications = classify_unintended_keys(
            states,
            current_target_keys=info["target_keys"],
            previous_target_keys=info.get("previous_target_keys", ()),
            future_target_keys=info.get("future_target_keys", ()),
            press_threshold=PRESS_THRESHOLD,
        )
        classifications_seen.extend(classifications)
        pressed_keys.update(int(k) for k in info["pressed_keys"])
        target_vectors.append(binary_key_vector(info["target_keys"]))
        pressed_vectors.append(binary_key_vector(info["pressed_keys"]))
        max_target = max(max_target, float(info["target_key_state"]))
        max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
        for key, value in info["reward_components"].items():
            reward_components[key].append(float(value))
        if render_dir is not None:
            frames.append(env.env.physics.render(height=480, width=640, camera_id=0))
            row = trace_row(
                step_index=step,
                frame_index=len(frames) - 1,
                simulation_time=step * dt,
                info=info,
                action=action,
                states=states,
                fingertips=env.fingertip_positions(),
                reward=reward,
            )
            row["sequence_id"] = sequence_id
            row["model_id"] = finalist.model_id
            row["rollout_seed"] = seed
            rows.append(row)
        step += 1
    if render_dir is None:
        # Build compact trace-like rows for the classifier without fingertip/render cost.
        # Re-run is intentionally avoided; repeated evaluation stays single-pass.
        rows = []
        # The classifier can work from aggregate fields below when no per-step trace is present.
    target_keys = {pitch - 21 for pitch in set(pitches)}
    pressed_metrics = pressed_key_metrics(target_keys, pressed_keys)
    timestep_metrics = timestep_key_metrics(target_vectors, pressed_vectors)
    unintended_values = [item.value for item in classifications_seen if item.value >= SOFT_THRESHOLD]
    category_counts = Counter(item.category for item in classifications_seen if item.is_pressed)
    action_saturation = float(np.mean(np.abs(np.asarray(actions)) >= ACTION_SATURATION_THRESHOLD)) if actions else 0.0
    if rows:
        labels = failure_labels_from_trace(
            pitches=pitches,
            rows=rows,
            pressed_keys=pressed_keys,
            timestep_f1=timestep_metrics.f1,
            action_saturation=action_saturation,
        )
    else:
        labels = aggregate_failure_labels(
            pitches=pitches,
            pressed_keys=pressed_keys,
            timestep_f1=timestep_metrics.f1,
            action_saturation=action_saturation,
            category_counts=category_counts,
        )
    metrics = {
        "model_id": finalist.model_id,
        "condition_id": finalist.condition_id,
        "algorithm": finalist.algorithm,
        "reward_profile": finalist.reward_profile,
        "checkpoint_step": finalist.checkpoint_step,
        "sequence_id": sequence_id,
        "sequence_group": sequence_group(sequence_id),
        "sequence_pitches": json.dumps(list(pitches)),
        "rollout_seed": seed,
        "policy_mode": "deterministic" if deterministic else "stochastic",
        "step_count": step,
        "control_timestep": dt,
        "pressed_keys": json.dumps(sorted(pressed_keys)),
        "pressed_midi_pitches": json.dumps([key_to_midi(k) for k in sorted(pressed_keys)]),
        "target_keys": json.dumps(sorted(target_keys)),
        "target_midi_pitches": json.dumps(sorted(set(pitches))),
        "pressed_key_precision": pressed_metrics.precision,
        "pressed_key_recall": pressed_metrics.recall,
        "pressed_key_f1": pressed_metrics.f1,
        "timestep_precision": timestep_metrics.precision,
        "timestep_recall": timestep_metrics.recall,
        "timestep_f1": timestep_metrics.f1,
        "integrated_unintended_travel": float(sum(unintended_values)),
        "wrong_key_threshold_crossings": len(pressed_keys - target_keys),
        "transition_completion": float(target_keys.issubset(pressed_keys)),
        "target_onset_success": float(bool(target_keys.intersection(pressed_keys))),
        "target_release_success": float(category_counts["previous_note_late_release"] == 0),
        "max_unintended_activation": max_unintended,
        "max_target_activation": max_target,
        "shaped_return": total_reward,
        "native_reward_sum": native_reward,
        "action_saturation_fraction": action_saturation,
        "failure_labels": json.dumps(labels),
        "strict_outcome": strict_outcome(target_keys, pressed_keys, max_target, max_unintended),
    }
    return metrics, rows, frames


def aggregate_failure_labels(
    *,
    pitches: tuple[int, ...],
    pressed_keys: set[int],
    timestep_f1: float,
    action_saturation: float,
    category_counts: Counter,
) -> list[str]:
    target_keys = [pitch - 21 for pitch in pitches]
    unique_targets = set(target_keys)
    labels: set[str] = set()
    if not unique_targets.intersection(pressed_keys):
        labels.add("target_never_pressed")
    if target_keys and target_keys[0] not in pressed_keys:
        labels.add("first_target_missed")
    if len(target_keys) >= 2 and target_keys[1] not in pressed_keys:
        labels.add("second_target_missed")
    if len(unique_targets) > 1 and not unique_targets.issubset(pressed_keys):
        labels.add("transition_incomplete")
    wrong_keys = pressed_keys - unique_targets
    if wrong_keys:
        neighbours = {target + delta for target in unique_targets for delta in (-1, 1)}
        if wrong_keys & neighbours:
            labels.add("neighbouring_wrong_key_press")
        if wrong_keys - neighbours:
            labels.add("unrelated_wrong_key_press")
        if len(wrong_keys) >= 2:
            labels.add("multiple_unintended_keys")
    if category_counts["previous_note_late_release"]:
        labels.add("previous_target_not_released")
    if category_counts["future_note_early_activation"]:
        labels.add("early_target_activation")
    if timestep_f1 < 0.5 and unique_targets.issubset(pressed_keys):
        labels.add("late_target_activation")
    if action_saturation >= ACTION_SATURATION_FRACTION_THRESHOLD:
        labels.add("action_saturation")
    return sorted(labels or {"none"})


def write_reference_rollout(
    *,
    finalist: Finalist,
    policy: PolicyAdapter,
    sequence_id: str,
    pitches: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    out = AUDIT_DIR / "reference_rollouts" / finalist.model_id / sequence_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics, rows, frames = rollout_once(
        finalist=finalist,
        policy=policy,
        sequence_id=sequence_id,
        pitches=pitches,
        seed=seed,
        deterministic=True,
        midi_dir=out,
        render_dir=out,
    )
    fps = max(1, int(round(1.0 / float(metrics["control_timestep"]))))
    video_path = out / "rollout.mp4"
    trace_path = out / "trace.csv"
    plot_path = out / "diagnostic_plot.png"
    imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)
    write_trace(trace_path, rows)
    write_plot(plot_path, rows)
    summary = {
        **metrics,
        "render_succeeded": True,
        "video_path": str(video_path),
        "trace_path": str(trace_path),
        "plot_path": str(plot_path),
        "frame_count": len(frames),
        "trace_rows": len(rows),
        "fps": fps,
    }
    write_json(out / "summary.json", summary)
    return summary


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


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def summarize_repeated(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    metric_cols = [
        "pressed_key_precision",
        "pressed_key_recall",
        "pressed_key_f1",
        "timestep_f1",
        "integrated_unintended_travel",
        "wrong_key_threshold_crossings",
        "transition_completion",
        "target_onset_success",
        "target_release_success",
        "max_unintended_activation",
        "action_saturation_fraction",
    ]
    aggregations = ["mean", "std", "min", "max", "median"]
    seq = df.groupby(["model_id", "condition_id", "algorithm", "reward_profile", "policy_mode", "sequence_id", "sequence_group"])[metric_cols].agg(aggregations)
    seq.columns = ["_".join(col).strip() for col in seq.columns.to_flat_index()]
    seq = seq.reset_index()
    seq["rollout_count"] = df.groupby(["model_id", "condition_id", "algorithm", "reward_profile", "policy_mode", "sequence_id", "sequence_group"]).size().values
    model = df.groupby(["model_id", "condition_id", "algorithm", "reward_profile", "policy_mode"])[metric_cols].agg(aggregations)
    model.columns = ["_".join(col).strip() for col in model.columns.to_flat_index()]
    model = model.reset_index()
    model["rollout_count"] = df.groupby(["model_id", "condition_id", "algorithm", "reward_profile", "policy_mode"]).size().values
    failure_rows = []
    for _, row in df.iterrows():
        labels = json.loads(row["failure_labels"])
        for label in labels:
            failure_rows.append({**{k: row[k] for k in ["model_id", "condition_id", "policy_mode", "sequence_id", "sequence_group"]}, "failure_label": label})
    failures = pd.DataFrame(failure_rows)
    if not failures.empty:
        failures = failures.groupby(["model_id", "condition_id", "policy_mode", "sequence_id", "sequence_group", "failure_label"]).size().reset_index(name="count")
        counts = df.groupby(["model_id", "condition_id", "policy_mode", "sequence_id", "sequence_group"]).size().reset_index(name="rollout_count")
        failures = failures.merge(counts, on=["model_id", "condition_id", "policy_mode", "sequence_id", "sequence_group"])
        failures["frequency"] = failures["count"] / failures["rollout_count"]
    return seq, model, failures


def plot_model_aggregates(seq_summary: pd.DataFrame) -> None:
    plot_dir = AUDIT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    deterministic = seq_summary[seq_summary["policy_mode"] == "deterministic"].copy()
    for model_id, group in deterministic.groupby("model_id"):
        rows = group.sort_values("sequence_id").to_dict("records")
        width = 1600
        height = 900
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        metrics = [
            ("F1", "pressed_key_f1_mean", (30, 90, 180), 0.0, 1.0),
            ("timestep F1", "timestep_f1_mean", (50, 140, 70), 0.0, 1.0),
            ("wrong", "wrong_key_threshold_crossings_mean", (180, 70, 30), 0.0, 3.0),
            ("unintended", "integrated_unintended_travel_mean", (130, 60, 150), 0.0, 8.0),
            ("transition success", "transition_completion_mean", (60, 150, 150), 0.0, 1.0),
            ("F1 std", "pressed_key_f1_std", (120, 120, 120), 0.0, 0.5),
        ]
        chart_w = width - 280
        bar_h = 9
        y = 35
        draw.text((20, 10), model_id, fill=(0, 0, 0))
        for row in rows:
            draw.text((20, y), row["sequence_id"][:28], fill=(0, 0, 0))
            x0 = 260
            for label, col, color, lo, hi in metrics:
                val = 0.0 if pd.isna(row.get(col)) else float(row.get(col, 0.0))
                frac = max(0.0, min(1.0, (val - lo) / max(1e-9, hi - lo)))
                draw.rectangle((x0, y, x0 + 160, y + bar_h), outline=(220, 220, 220))
                draw.rectangle((x0, y, x0 + int(160 * frac), y + bar_h), fill=color)
                x0 += 180
            y += 20
        legend_y = height - 70
        x = 260
        for label, _, color, _, _ in metrics:
            draw.rectangle((x, legend_y, x + 20, legend_y + 12), fill=color)
            draw.text((x + 25, legend_y - 2), label, fill=(0, 0, 0))
            x += 190
        img.save(plot_dir / f"{model_id}_aggregate_metrics.png")


def make_manual_review(reference_rows: list[dict[str, Any]], repeated_rows: list[dict[str, Any]]) -> None:
    review_dir = AUDIT_DIR / "manual_review"
    video_dir = review_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    ref = pd.DataFrame(reference_rows)
    selections: list[tuple[str, pd.Series]] = []
    for model_id, group in ref.groupby("model_id"):
        worst = group.sort_values(["pressed_key_f1", "timestep_f1", "integrated_unintended_travel"], ascending=[True, True, False]).iloc[0]
        best = group.sort_values(["pressed_key_f1", "timestep_f1", "integrated_unintended_travel"], ascending=[False, False, True]).iloc[0]
        selections.append((f"worst rollout for {model_id}", worst))
        selections.append((f"best rollout for {model_id}", best))
    winner = ref[ref["model_id"] == "droq_sensitive_v1_800k_frozen"]
    for group_name, reason in [
        ("trained_transition", "winner worst trained transition"),
        ("heldout_transition", "winner worst held-out sequence"),
        ("composition_probe", "winner worst composition probe"),
    ]:
        sub = winner[winner["sequence_group"] == group_name]
        if not sub.empty:
            row = sub.sort_values(["pressed_key_f1", "timestep_f1", "integrated_unintended_travel"], ascending=[True, True, False]).iloc[0]
            selections.append((reason, row))
    unrelated = ref[ref["failure_labels"].str.contains("unrelated_wrong_key_press", regex=False)]
    if not unrelated.empty:
        selections.append(("unrelated wrong-key press example", unrelated.iloc[0]))
    # Matched sequence where finalists differ strongly by F1.
    pivot = ref.pivot(index="sequence_id", columns="model_id", values="pressed_key_f1")
    if not pivot.empty:
        spread = (pivot.max(axis=1) - pivot.min(axis=1)).sort_values(ascending=False)
        if not spread.empty:
            seq_id = spread.index[0]
            for _, row in ref[ref["sequence_id"] == seq_id].iterrows():
                selections.append((f"side-by-side behaviour spread on {seq_id}", row))
    seen = set()
    lines = ["# Manual Review Index", ""]
    for reason, row in selections:
        key = (row["model_id"], row["sequence_id"], row["rollout_seed"], reason)
        if key in seen:
            continue
        seen.add(key)
        src = Path(row["video_path"])
        link_name = f"{len(seen):02d}_{row['model_id']}_{row['sequence_id']}.mp4"
        dest = video_dir / link_name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(src)
        except OSError:
            shutil.copy2(src, dest)
        lines.extend(
            [
                f"## {len(seen)}. {reason}",
                "",
                f"- model: `{row['model_id']}`",
                f"- sequence: `{row['sequence_id']}`",
                f"- rollout seed: `{row['rollout_seed']}`",
                f"- metrics: F1 `{row['pressed_key_f1']:.3f}`, timestep F1 `{row['timestep_f1']:.3f}`, unintended `{row['integrated_unintended_travel']:.3f}`, wrong `{row['wrong_key_threshold_crossings']}`",
                f"- failure labels: `{row['failure_labels']}`",
                f"- MP4: `{src}`",
                f"- trace: `{row['trace_path']}`",
                f"- plot: `{row['plot_path']}`",
                f"- review copy/link: `{dest}`",
                "",
            ]
        )
    (AUDIT_DIR / "manual_review_index.md").write_text("\n".join(lines), encoding="utf-8")


def diagnose_smoke_rollout() -> None:
    trace_path = FROZEN_RELEASE / "validation/smoke_rollout/trained_73_72/trace.csv"
    df = pd.read_csv(trace_path)
    unintended_rows = df[df["wrong_key_threshold_crossing_events"].astype(str) != "[]"]
    lines = [
        "# Smoke Rollout Diagnosis",
        "",
        "The existing smoke trace uses RoboPianist piano-key indices in `pressed_keys`, not MIDI pitch numbers. RoboPianist maps `key_index = midi_pitch - 21`, so key 52 is MIDI 73 (C#5), key 51 is MIDI 72 (C5), and key 42 is MIDI 63 (D#4).",
        "",
        "For `trained_73_72`, the intended MIDI sequence is 73 then 72, so the intended key indices are 52 then 51.",
        "",
        "Observed trace evidence:",
        "",
    ]
    for _, row in unintended_rows.iterrows():
        events = json.loads(row["wrong_key_threshold_crossing_events"])
        lines.append(
            f"- step {int(row['step_index'])}, t={float(row['simulation_time']):.2f}s, target `{row['intended_midi_pitches']}`, pressed `{row['pressed_keys']}`, events `{events}`"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Key 52 was the intended first target and remained pressed into the release gap at step 5, classified as `previous_note_late_release`.",
            "- Key 42 was not a MIDI label; it is piano-key index 42, i.e. MIDI 63/D#4. It was an unrelated unintended press during the second target window.",
            "- The trace therefore reflects real controller behaviour: the model missed the second intended key 51 and produced an unrelated low-register contact. It is not a key-index/MIDI conversion bug.",
            "- Future audit traces add explicit `pressed_key_indices`, `pressed_midi_pitches`, and `intended_key_indices` fields to reduce ambiguity. The original smoke trace is left unchanged.",
        ]
    )
    (AUDIT_DIR / "smoke_rollout_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def write_audit_report(reference_rows: list[dict[str, Any]], repeated_rows: list[dict[str, Any]], seq_summary: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    ref = pd.DataFrame(reference_rows)
    det_model = model_summary[model_summary["policy_mode"] == "deterministic"].sort_values("pressed_key_f1_mean", ascending=False)
    winner = det_model.iloc[0] if not det_model.empty else None
    lines = [
        "# Five-Note Finalist Rollout Audit",
        "",
        "## Scope",
        "",
        f"- models audited: {ref['model_id'].nunique() if not ref.empty else 0}",
        f"- sequences audited: {ref['sequence_id'].nunique() if not ref.empty else 0}",
        f"- reference rollout videos: {len(ref)}",
        f"- repeated rollout rows: {len(repeated_rows)}",
        "",
        "## Failure Taxonomy",
        "",
        f"- press threshold: {PRESS_THRESHOLD}",
        f"- unintended soft threshold: {SOFT_THRESHOLD}",
        f"- action saturation label: fraction of normalized actions with abs(action) >= {ACTION_SATURATION_THRESHOLD} is at least {ACTION_SATURATION_FRACTION_THRESHOLD}",
        f"- target press too brief: target key crosses press threshold for fewer than {TARGET_BRIEF_PRESS_STEPS} control steps",
        "- multiple labels may apply to a rollout.",
        "",
        "## Deterministic Repeated Evaluation Summary",
        "",
    ]
    if not det_model.empty:
        cols = [
            "model_id",
            "pressed_key_precision_mean",
            "pressed_key_recall_mean",
            "pressed_key_f1_mean",
            "timestep_f1_mean",
            "integrated_unintended_travel_mean",
            "wrong_key_threshold_crossings_mean",
            "transition_completion_mean",
            "max_unintended_activation_mean",
        ]
        lines.append(markdown_table(det_model[cols].to_dict("records"), cols))
        lines.append("")
    lines.extend(
        [
            "## Answers",
            "",
            f"- Balanced winner status: `{winner['model_id'] if winner is not None else 'unknown'}` has the highest mean deterministic pressed-key F1 in this audit table. The final recommendation should still account for cleanliness and timing trade-offs rather than one scalar.",
            "- SAC-sensitive mechanical reliability: directly observable as lower wrong-key crossings and unintended travel where its targets activate, but it is also more conservative and misses second notes more often.",
            "- SAC-original timing: strong timestep F1 in the factorial analysis; this audit quantifies whether that remains stable across reset seeds in `per_model_summary.csv`.",
            "- Dominant failure mechanisms are reported by frequency in `repeated_evaluation/failure_frequency.csv`; labels distinguish missed targets, release failures, unintended contacts and action saturation.",
            "- DroQ-sensitive dominant failure mechanism should be read from the failure-frequency table, with the smoke anomaly showing a concrete unrelated wrong-key contact on `trained_73_72`.",
            "- Determinism/repeatability: repeated deterministic-reset variance is summarized in `per_sequence_summary.csv`; if standard deviations are zero or near-zero, reset seeds do not materially vary the fixed-MIDI environment.",
            "",
            "## Evidence Boundaries",
            "",
            "- Direct trace evidence: reference rollout traces and videos under `reference_rollouts/`.",
            "- Aggregate repeated-evaluation evidence: CSVs under `repeated_evaluation/`.",
            "- Visual/manual interpretation: `manual_review_index.md` and aggregate plots.",
            "- Tentative hypotheses: any explanation of why a finger caused a contact is tentative unless backed by frame/trace inspection.",
            "",
            "## Targeted Intervention",
            "",
            "The single most justified intervention is to address release/transition control explicitly in the learning objective or policy conditioning, because the finalist traces show that target hits alone are insufficient: missed second notes, previous-target late release and unintended contacts remain the key behavioural bottlenecks.",
        ]
    )
    (AUDIT_DIR / "finalist_rollout_audit.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit(*, repeat_count: int, include_stochastic: bool, render: bool) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    sequences = audit_sequences()
    finalists = finalist_definitions()
    seeds = deterministic_seed_list(repeat_count)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sequence_count": len(sequences),
        "sequences": {name: list(pitches) for name, pitches in sequences.items()},
        "finalists": [
            {
                "model_id": f.model_id,
                "condition_id": f.condition_id,
                "algorithm": f.algorithm,
                "reward_profile": f.reward_profile,
                "checkpoint_step": f.checkpoint_step,
                "checkpoint_path": str(f.checkpoint_path),
                "checkpoint_sha256": sha256_file(f.checkpoint_path),
            }
            for f in finalists
        ],
        "repeat_count": repeat_count,
        "seeds": seeds,
        "include_stochastic": include_stochastic,
        "valid_variation": "reset seed varied through env.reset(seed=...); fixed MIDI benchmark sequences are otherwise deterministic.",
    }
    write_json(AUDIT_DIR / "audit_manifest.json", metadata)
    reference_rows: list[dict[str, Any]] = []
    repeated_rows: list[dict[str, Any]] = []
    for finalist in finalists:
        policy = load_policy(finalist)
        for sequence_id, pitches in sequences.items():
            if render:
                summary = write_reference_rollout(
                    finalist=finalist,
                    policy=policy,
                    sequence_id=sequence_id,
                    pitches=pitches,
                    seed=13,
                )
                reference_rows.append(summary)
            for mode in (["deterministic", "stochastic"] if include_stochastic else ["deterministic"]):
                deterministic = mode == "deterministic"
                for seed in seeds:
                    metrics, _, _ = rollout_once(
                        finalist=finalist,
                        policy=policy,
                        sequence_id=sequence_id,
                        pitches=pitches,
                        seed=seed,
                        deterministic=deterministic,
                        midi_dir=AUDIT_DIR / "repeated_evaluation/eval_midi" / finalist.model_id,
                        render_dir=None,
                    )
                    repeated_rows.append(metrics)
    write_csv(AUDIT_DIR / "reference_rollout_metrics.csv", reference_rows)
    rep_dir = AUDIT_DIR / "repeated_evaluation"
    write_csv(rep_dir / "per_rollout_metrics.csv", repeated_rows)
    seq_summary, model_summary, failures = summarize_repeated(repeated_rows)
    seq_summary.to_csv(rep_dir / "per_sequence_summary.csv", index=False)
    model_summary.to_csv(rep_dir / "per_model_summary.csv", index=False)
    failures.to_csv(rep_dir / "failure_frequency.csv", index=False)
    plot_model_aggregates(seq_summary)
    if render:
        make_manual_review(reference_rows, repeated_rows)
    diagnose_smoke_rollout()
    write_audit_report(reference_rows, repeated_rows, seq_summary, model_summary)
    return {
        "models": len(finalists),
        "sequences": len(sequences),
        "reference_rollouts": len(reference_rows),
        "repeated_rollouts": len(repeated_rows),
        "all_videos_rendered": bool(render and len(reference_rows) == len(finalists) * len(sequences)),
        "audit_dir": str(AUDIT_DIR),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument("--include-stochastic", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    result = run_audit(
        repeat_count=args.repeat_count,
        include_stochastic=args.include_stochastic,
        render=not args.skip_render,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
