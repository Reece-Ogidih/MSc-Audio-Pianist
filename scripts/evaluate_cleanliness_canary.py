#!/usr/bin/env python
"""Evaluate local DroQ cleanliness-canary checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ala_pianist.evaluation import binary_key_vector, pressed_key_metrics, timestep_key_metrics
from ala_pianist.evaluation.unintended import classify_unintended_keys
from ala_pianist.music import assign_right_hand_fingering, sequence_timing_from_profile, write_sequence_midi
from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv
from evaluate_general_one_hand_policy import reward_config_from_profile


ROOT = Path("/home/reece_dev/msc-audio-pianist")
SEQUENCES = {
    "single_73": [73],
    "single_74": [74],
    "single_75": [75],
    "transition_73_75": [73, 75],
    "transition_75_73": [75, 73],
    "transition_74_75": [74, 75],
}
TRANSITIONS = {"transition_73_75", "transition_75_73", "transition_74_75"}


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)_steps\.pt$", path.name)
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def discover_checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("**/checkpoint_*_steps.pt"), key=checkpoint_step)


def write_eval_midi(out_dir: Path, name: str, pitches: list[int]) -> Path:
    return write_sequence_midi(
        pitches,
        out_dir / "eval_midi" / f"{name}_{'_'.join(map(str, pitches))}.mid",
        midi_min=min(pitches),
        midi_max=max(pitches),
        timing=sequence_timing_from_profile("aligned"),
        fingering_fn=assign_right_hand_fingering,
        title=f"cleanliness canary {name}",
    )


def evaluate_checkpoint(path: Path, out_dir: Path, *, soft_threshold: float, press_threshold: float):
    policy = DroQPolicy.load(path, device="cpu")
    cfg = reward_config_from_profile("transition_cleanup_sensitive_v1")
    rows = []
    reward_rows = []
    trajectories = {}
    for name, pitches in SEQUENCES.items():
        midi_path = write_eval_midi(out_dir, name, pitches)
        env = GeneralOneHandGoalEnv(
            midi_path=midi_path,
            midi_min=min(pitches),
            midi_max=max(pitches),
            seed=23,
            lookahead=1,
            horizon_steps=96,
            action_mode="direct",
            action_repeat=1,
            reward_config=cfg,
        )
        obs, info = env.reset(seed=23)
        total = 0.0
        native = 0.0
        max_target = 0.0
        max_unintended = 0.0
        pressed_keys = set()
        target_vectors = []
        pressed_vectors = []
        soft_count = 0
        press_count = 0
        wrong_press_count = 0
        integrated = 0.0
        late_release = 0.0
        early_activation = 0.0
        reward_components = defaultdict(list)
        trajectory = []
        for step in range(env.horizon_steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            states = env.piano_key_states()
            classifications = classify_unintended_keys(
                states,
                current_target_keys=info["target_keys"],
                previous_target_keys=info.get("previous_target_keys", ()),
                future_target_keys=info.get("future_target_keys", ()),
                press_threshold=press_threshold,
            )
            for item in classifications:
                if item.value >= soft_threshold:
                    soft_count += 1
                if item.value >= press_threshold:
                    press_count += 1
                integrated += max(0.0, item.value - soft_threshold)
                if item.category == "previous_note_late_release":
                    late_release += max(0.0, item.value - soft_threshold)
                if item.category == "future_note_early_activation":
                    early_activation += max(0.0, item.value - soft_threshold)
            total += float(reward)
            native += float(info["native_reward"])
            max_target = max(max_target, float(info["target_key_state"]))
            max_unintended = max(max_unintended, float(info["max_unintended_key_state"]))
            pressed_keys.update(info["pressed_keys"])
            target_vectors.append(binary_key_vector(info["target_keys"]))
            pressed_vectors.append(binary_key_vector(info["pressed_keys"]))
            for key, value in info["reward_components"].items():
                reward_components[key].append(float(value))
            trajectory.append(
                {
                    "step": step,
                    "key_states": [round(float(v), 6) for v in states],
                    "target_keys": list(info["target_keys"]),
                    "pressed_keys": list(info["pressed_keys"]),
                }
            )
            if terminated or truncated:
                break
        target_keys = {pitch - 21 for pitch in pitches}
        wrong_pressed = set(pressed_keys) - target_keys
        wrong_press_count = len(wrong_pressed)
        pressed = pressed_key_metrics(target_keys, pressed_keys)
        timestep = timestep_key_metrics(target_vectors, pressed_vectors)
        step = checkpoint_step(path)
        rows.append(
            {
                "checkpoint_step": step,
                "checkpoint_path": str(path),
                "sequence_name": name,
                "sequence_pitches": json.dumps(pitches),
                "pressed_keys": json.dumps(sorted(int(k) for k in pressed_keys)),
                "pressed_key_precision": pressed.precision,
                "pressed_key_recall": pressed.recall,
                "pressed_key_f1": pressed.f1,
                "timestep_precision": timestep.precision,
                "timestep_recall": timestep.recall,
                "timestep_f1": timestep.f1,
                "max_target_key_state": max_target,
                "max_unintended_key_state": max_unintended,
                "integrated_unintended_travel": integrated,
                "timesteps_above_soft_threshold": soft_count,
                "timesteps_above_press_threshold": press_count,
                "wrong_key_press_count": wrong_press_count,
                "late_release_duration": late_release,
                "early_activation_duration": early_activation,
                "shaped_return": total,
                "native_reward_sum": native,
            }
        )
        for key, values in reward_components.items():
            arr = np.asarray(values, dtype=float)
            reward_rows.append(
                {
                    "checkpoint_step": step,
                    "sequence_name": name,
                    "component": key,
                    "mean": float(np.mean(arr)),
                    "max": float(np.max(arr)),
                    "total": float(np.sum(arr)),
                }
            )
        if name in TRANSITIONS:
            trajectories[(step, name)] = trajectory
    return rows, reward_rows, trajectories


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["checkpoint_step"]].append(row)
    summaries = []
    for step, items in grouped.items():
        summaries.append(
            {
                "checkpoint_step": step,
                "mean_pressed_key_f1": float(np.mean([r["pressed_key_f1"] for r in items])),
                "mean_timestep_f1": float(np.mean([r["timestep_f1"] for r in items])),
                "mean_max_unintended": float(np.mean([r["max_unintended_key_state"] for r in items])),
                "mean_integrated_unintended_travel": float(np.mean([r["integrated_unintended_travel"] for r in items])),
                "mean_shaped_return": float(np.mean([r["shaped_return"] for r in items])),
            }
        )
    return sorted(summaries, key=lambda r: r["checkpoint_step"])


def svg_plot(path: Path, title: str, points: list[tuple[float, float]], ylabel: str) -> None:
    width, height = 720, 420
    left, right, top, bottom = 70, 30, 50, 60
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymax += 1.0
    def sx(x): return left + (x - xmin) / max(1e-6, xmax - xmin) * (width - left - right)
    def sy(y): return top + (ymax - y) / max(1e-6, ymax - ymin) * (height - top - bottom)
    coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
                f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
                f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{coords}"/>',
                f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="13">Checkpoint step</text>',
                f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def svg_key_state_plot(path: Path, title: str, trajectory: list[dict[str, Any]]) -> None:
    width, height = 840, 460
    left, right, top, bottom = 70, 160, 50, 60
    key_set = set()
    for row in trajectory:
        key_set.update(int(key) for key in row["target_keys"])
        key_set.update(int(key) for key in row["pressed_keys"])
    if not key_set:
        key_set = {52, 53, 54}
    keys = sorted(key_set)
    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    xmax = max(1, len(trajectory) - 1)

    def sx(step):
        return left + float(step) / xmax * (width - left - right)

    def sy(value):
        return top + (1.0 - float(value)) * (height - top - bottom)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{sy(0.5):.1f}" x2="{width-right}" y2="{sy(0.5):.1f}" stroke="#9ca3af" stroke-dasharray="4,4"/>',
    ]
    for index, key in enumerate(keys):
        color = palette[index % len(palette)]
        points = " ".join(
            f"{sx(row['step']):.1f},{sy(row['key_states'][key]):.1f}"
            for row in trajectory
        )
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>')
        legend_y = top + 24 + index * 22
        lines.extend(
            [
                f'<line x1="{width-right+20}" y1="{legend_y}" x2="{width-right+48}" y2="{legend_y}" stroke="{color}" stroke-width="2.2"/>',
                f'<text x="{width-right+56}" y="{legend_y+4}" font-family="Arial" font-size="13">key {key}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="13">Timestep</text>',
            f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">Key state</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--soft-threshold", type=float, default=0.2)
    parser.add_argument("--press-threshold", type=float, default=0.5)
    args = parser.parse_args()
    out = Path(args.output_dir)
    checkpoints = discover_checkpoints(Path(args.run_dir))
    all_rows = []
    all_reward_rows = []
    all_trajectories = {}
    for checkpoint in checkpoints:
        rows, reward_rows, trajectories = evaluate_checkpoint(
            checkpoint,
            out,
            soft_threshold=args.soft_threshold,
            press_threshold=args.press_threshold,
        )
        all_rows.extend(rows)
        all_reward_rows.extend(reward_rows)
        all_trajectories.update(trajectories)
    summaries = summarize(all_rows)
    write_csv(out / "per_checkpoint_per_sequence_metrics.csv", all_rows)
    write_csv(out / "reward_component_summary.csv", all_reward_rows)
    write_csv(out / "per_checkpoint_summary.csv", summaries)
    if summaries:
        svg_plot(
            out / "plots" / "mean_f1_over_checkpoints.svg",
            "Canary mean pressed-key F1",
            [(row["checkpoint_step"], row["mean_pressed_key_f1"]) for row in summaries],
            "Mean F1",
        )
        svg_plot(
            out / "plots" / "mean_integrated_unintended.svg",
            "Canary integrated unintended travel",
            [(row["checkpoint_step"], row["mean_integrated_unintended_travel"]) for row in summaries],
            "Integrated unintended",
        )
    trajectory_plots = []
    for (step, name), trajectory in sorted(all_trajectories.items()):
        path = out / "plots" / "key_state_trajectories" / f"checkpoint_{step}_{name}.svg"
        svg_key_state_plot(path, f"{name} key-state trajectory at {step} steps", trajectory)
        trajectory_plots.append(str(path))
    payload = {
        "checkpoints": [str(path) for path in checkpoints],
        "summary": summaries,
        "outputs": {
            "per_checkpoint_per_sequence_metrics": str(out / "per_checkpoint_per_sequence_metrics.csv"),
            "per_checkpoint_summary": str(out / "per_checkpoint_summary.csv"),
            "reward_component_summary": str(out / "reward_component_summary.csv"),
            "trajectory_plots": trajectory_plots,
        },
    }
    (out / "canary_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["outputs"], indent=2))


if __name__ == "__main__":
    main()
