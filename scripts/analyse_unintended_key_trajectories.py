#!/usr/bin/env python
"""Analyse per-timestep unintended piano-key trajectories for trained policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from ala_pianist.evaluation.unintended import classify_unintended_keys
from ala_pianist.music import assign_right_hand_fingering, sequence_timing_from_profile, write_sequence_midi
from ala_pianist.music.sequence_generation import note_windows
from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv
from evaluate_general_one_hand_policy import reward_config_from_profile


ROOT = Path("/home/reece_dev/msc-audio-pianist")
DEFAULT_OUT = ROOT / "experiments" / "general_one_hand" / "droq" / "evaluation" / "unintended_key_trajectories"

DEFAULT_MODELS = {
    "droq_300k": (
        "droq",
        ROOT / "experiments/general_one_hand/droq/checkpoints/droq_stage3c_fair_1m_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_seed13_1000000/checkpoint_300000_steps.pt",
    ),
    "droq_700k": (
        "droq",
        ROOT / "artifacts/hex_runs/droq_stage3c_fair_1m_20260720T160108Z/output/checkpoints/droq_stage3c_fair_hexcloud_resume_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_seed13_1000000/checkpoint_700000_steps.pt",
    ),
    "droq_900k": (
        "droq",
        ROOT / "artifacts/hex_runs/droq_stage3c_fair_1m_20260720T160108Z/output/checkpoints/droq_stage3c_fair_hexcloud_resume_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_seed13_1000000/checkpoint_900000_steps.pt",
    ),
    "droq_1m": (
        "droq",
        ROOT / "artifacts/hex_runs/droq_stage3c_fair_1m_20260720T160108Z/output/checkpoints/droq_stage3c_fair_hexcloud_resume_droq_sequence_cleanup_lookahead1_directx1_transition_cleanup_seed13_1000000/checkpoint_1000000_steps.pt",
    ),
    "sac_stage3c": (
        "sac",
        ROOT / "experiments/general_one_hand/stage3c_adjacent_74_75_general_one_hand_sac_sequence_cleanup_pitches73-74-75_lookahead1_directx1_transition_cleanup_seed13_500000.zip",
    ),
}

SEQUENCES = {
    "single_73": [73],
    "single_74": [74],
    "single_75": [75],
    "transition_73_75": [73, 75],
    "transition_75_73": [75, 73],
    "transition_74_75": [74, 75],
    "probe_73_74": [73, 74],
    "probe_75_74": [75, 74],
    "return_73_75_73": [73, 75, 73],
}
PLOT_SEQUENCES = {"transition_73_75", "transition_75_73", "transition_74_75"}


def load_policy(kind: str, path: Path):
    if kind == "droq":
        return DroQPolicy.load(path, device="cpu")
    if kind == "sac":
        return SAC.load(path)
    raise ValueError(f"Unknown policy kind {kind!r}")


def write_eval_midi(out_dir: Path, sequence_name: str, pitches: list[int]) -> Path:
    timing = sequence_timing_from_profile("aligned")
    return write_sequence_midi(
        pitches,
        out_dir / "eval_midi" / f"{sequence_name}_{'_'.join(map(str, pitches))}.mid",
        midi_min=min(pitches),
        midi_max=max(pitches),
        timing=timing,
        fingering_fn=assign_right_hand_fingering,
        title=f"unintended diagnostic {sequence_name}",
    )


def contact_summary(env: GeneralOneHandGoalEnv) -> str:
    try:
        physics = env.env.physics
        contacts = []
        for idx in range(physics.data.ncon):
            contact = physics.data.contact[idx]
            geom1 = physics.model.id2name(contact.geom1, "geom") or str(contact.geom1)
            geom2 = physics.model.id2name(contact.geom2, "geom") or str(contact.geom2)
            text = f"{geom1}|{geom2}"
            if "finger" in text or "TH" in text or "FF" in text or "MF" in text or "RF" in text or "LF" in text:
                if "key" in text or "white" in text or "black" in text:
                    contacts.append(text)
        return ";".join(contacts[:8])
    except Exception:
        return ""


def nearest_fingertip(env: GeneralOneHandGoalEnv, key: int | None) -> dict[str, Any]:
    if key is None:
        return {}
    try:
        result = env.nearest_fingertip_to_key(int(key))
    except Exception:
        return {}
    if not result:
        return {}
    return {
        "nearest_fingertip": result.get("fingertip"),
        "nearest_fingertip_distance": result.get("distance"),
    }


def evaluate_model(
    *,
    model_label: str,
    kind: str,
    model_path: Path,
    out_dir: Path,
    soft_thresholds: tuple[float, ...],
    press_threshold: float,
    horizon_steps: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    policy = load_policy(kind, model_path)
    timestep_rows = []
    episode_rows = []
    trajectories: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for sequence_name, pitches in SEQUENCES.items():
        midi_path = write_eval_midi(out_dir, sequence_name, pitches)
        env = GeneralOneHandGoalEnv(
            midi_path=midi_path,
            midi_min=min(pitches),
            midi_max=max(pitches),
            seed=seed,
            lookahead=1,
            horizon_steps=horizon_steps,
            action_mode="direct",
            action_repeat=1,
            reward_config=reward_config_from_profile("transition_cleanup"),
        )
        obs, info = env.reset(seed=seed)
        windows = note_windows(pitches, timing=sequence_timing_from_profile("aligned"))
        target_key_set = {pitch - 21 for pitch in pitches}
        max_unintended = 0.0
        max_unintended_key = None
        category_counts = Counter()
        soft_counts = {str(threshold): 0 for threshold in soft_thresholds}
        press_count = 0
        integrated_unintended = 0.0
        pressed_union = set()
        rows_for_plot = []
        for step in range(horizon_steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            states = env.piano_key_states()
            target_keys = tuple(int(key) for key in info["target_keys"])
            previous_keys = tuple(int(key) for key in info.get("previous_target_keys", ()))
            future_keys = tuple(int(key) for key in info.get("future_target_keys", ()))
            classifications = classify_unintended_keys(
                states,
                current_target_keys=target_keys,
                previous_target_keys=previous_keys,
                future_target_keys=future_keys,
                press_threshold=press_threshold,
            )
            max_item = max(classifications, key=lambda item: item.value)
            max_unintended = max(max_unintended, max_item.value)
            if max_item.value >= max_unintended:
                max_unintended_key = max_item.key_index
            for item in classifications:
                if item.value > 0.0:
                    category_counts[item.category] += 1
                if item.is_pressed:
                    press_count += 1
                for threshold in soft_thresholds:
                    if item.value >= threshold:
                        soft_counts[str(threshold)] += 1
                integrated_unintended += max(0.0, item.value - min(soft_thresholds))
            pressed_union.update(int(key) for key in info["pressed_keys"])
            nearest = nearest_fingertip(env, max_item.key_index)
            row = {
                "model_label": model_label,
                "model_kind": kind,
                "model_path": str(model_path),
                "sequence_name": sequence_name,
                "sequence_pitches": json.dumps(pitches),
                "step": step,
                "time_seconds": step * 0.05,
                "current_target_keys": json.dumps(list(target_keys)),
                "previous_target_keys": json.dumps(list(previous_keys)),
                "future_target_keys": json.dumps(list(future_keys)),
                "pressed_keys": json.dumps([int(key) for key in info["pressed_keys"]]),
                "key_states": json.dumps([round(float(value), 6) for value in states]),
                "unintended_values": json.dumps(
                    {str(item.key_index): round(item.value, 6) for item in classifications if item.value > 0.0}
                ),
                "max_unintended_key": max_item.key_index,
                "max_unintended_value": max_item.value,
                "max_unintended_pressed": max_item.is_pressed,
                "max_unintended_category": max_item.category,
                "max_unintended_neighbouring": max_item.is_neighbouring_key,
                "soft_threshold_counts": json.dumps(soft_counts),
                "press_threshold_count_so_far": press_count,
                "note_windows": json.dumps(windows),
                "contact_summary": contact_summary(env),
                "reward_components": json.dumps(info.get("reward_components", {}), sort_keys=True),
                **nearest,
            }
            timestep_rows.append(row)
            rows_for_plot.append(row)
            if terminated or truncated:
                break
        episode_rows.append(
            {
                "model_label": model_label,
                "model_kind": kind,
                "sequence_name": sequence_name,
                "sequence_pitches": json.dumps(pitches),
                "target_keys": json.dumps(sorted(target_key_set)),
                "pressed_keys": json.dumps(sorted(pressed_union)),
                "max_unintended_key": max_unintended_key,
                "max_unintended_value": max_unintended,
                "category_counts": json.dumps(category_counts, sort_keys=True),
                "soft_threshold_counts": json.dumps(soft_counts, sort_keys=True),
                "press_threshold_count": press_count,
                "integrated_unintended_travel": integrated_unintended,
            }
        )
        trajectories[(model_label, sequence_name)] = rows_for_plot
    return timestep_rows, episode_rows, trajectories


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sequence_summary(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in episode_rows:
        grouped[row["sequence_name"]].append(row)
    summaries = []
    for sequence_name, rows in grouped.items():
        categories = Counter()
        for row in rows:
            categories.update(json.loads(row["category_counts"]))
        summaries.append(
            {
                "sequence_name": sequence_name,
                "episodes": len(rows),
                "mean_max_unintended": float(np.mean([row["max_unintended_value"] for row in rows])),
                "mean_integrated_unintended_travel": float(
                    np.mean([row["integrated_unintended_travel"] for row in rows])
                ),
                "category_counts": json.dumps(categories, sort_keys=True),
            }
        )
    return summaries


def svg_key_state_plot(path: Path, rows: list[dict[str, Any]], title: str, pitches: list[int]) -> None:
    width, height = 960, 520
    left, right, top, bottom = 70, 30, 55, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    times = [float(row["time_seconds"]) for row in rows]
    if not times:
        return
    keys = sorted({pitch - 21 for pitch in pitches} | {key for row in rows for key in json.loads(row["pressed_keys"])})
    for row in rows:
        max_key = row.get("max_unintended_key")
        if max_key != "":
            keys.append(int(max_key))
    keys = sorted(set(keys))
    colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2", "#4b5563", "#be123c"]
    x_min, x_max = min(times), max(times) or 1.0
    def sx(x): return left + (x - x_min) / max(1e-6, x_max - x_min) * plot_w
    def sy(y): return top + (1.0 - y) * plot_h
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{title}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#111827"/>',
    ]
    windows = json.loads(rows[0]["note_windows"])
    for window in windows:
        x1, x2 = sx(float(window["start_seconds"])), sx(float(window["end_seconds"]))
        lines.append(f'<rect x="{x1:.1f}" y="{top}" width="{max(1, x2-x1):.1f}" height="{plot_h}" fill="#dbeafe" opacity="0.28"/>')
        lines.append(f'<text x="{(x1+x2)/2:.1f}" y="{top+14}" text-anchor="middle" font-family="Arial" font-size="11">key {window["key_index"]}</text>')
    for i in range(6):
        y = i / 5
        yy = sy(y)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{y:.1f}</text>')
    for idx, key in enumerate(keys):
        points = []
        for row in rows:
            states = json.loads(row["key_states"])
            points.append((float(row["time_seconds"]), float(states[key])))
        color = colors[idx % len(colors)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{coords}"/>')
        lines.append(f'<text x="{left+plot_w-130}" y="{top+18+idx*17}" font-family="Arial" font-size="11" fill="{color}">key {key}</text>')
    lines.append(f'<text x="{left+plot_w/2}" y="{height-22}" text-anchor="middle" font-family="Arial" font-size="13">Time (s)</text>')
    lines.append('<text transform="translate(22 260) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">Normalised key state</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--soft-thresholds", default="0.2,0.5,0.75")
    parser.add_argument("--press-threshold", type=float, default=0.5)
    parser.add_argument("--horizon-steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--model", action="append", default=None, help="label:kind:path")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    soft_thresholds = tuple(float(part.strip()) for part in args.soft_thresholds.split(",") if part.strip())
    if args.model:
        models = {}
        for raw in args.model:
            label, kind, path = raw.split(":", 2)
            models[label] = (kind, Path(path))
    else:
        models = DEFAULT_MODELS

    timestep_rows = []
    episode_rows = []
    all_trajectories = {}
    for label, (kind, path) in models.items():
        if not path.exists():
            print(f"skipping_missing_model={label} path={path}")
            continue
        print(f"evaluating_model={label}")
        rows, summaries, trajectories = evaluate_model(
            model_label=label,
            kind=kind,
            model_path=path,
            out_dir=out_dir,
            soft_thresholds=soft_thresholds,
            press_threshold=args.press_threshold,
            horizon_steps=args.horizon_steps,
            seed=args.seed,
        )
        timestep_rows.extend(rows)
        episode_rows.extend(summaries)
        all_trajectories.update(trajectories)

    sequence_rows = sequence_summary(episode_rows)
    write_csv(out_dir / "per_timestep_unintended.csv", timestep_rows)
    write_csv(out_dir / "per_episode_unintended_summary.csv", episode_rows)
    write_csv(out_dir / "per_sequence_unintended_summary.csv", sequence_rows)

    for (model_label, sequence_name), rows in all_trajectories.items():
        if sequence_name not in PLOT_SEQUENCES:
            continue
        svg_key_state_plot(
            out_dir / "plots" / f"{model_label}_{sequence_name}_key_states.svg",
            rows,
            f"{model_label} {sequence_name} key states",
            SEQUENCES[sequence_name],
        )

    category_counts = Counter()
    for row in episode_rows:
        category_counts.update(json.loads(row["category_counts"]))
    payload = {
        "models": {label: {"kind": kind, "path": str(path)} for label, (kind, path) in models.items()},
        "sequences": SEQUENCES,
        "soft_thresholds": soft_thresholds,
        "press_threshold": args.press_threshold,
        "category_counts": dict(category_counts),
        "outputs": {
            "per_timestep_unintended": str(out_dir / "per_timestep_unintended.csv"),
            "per_episode_unintended_summary": str(out_dir / "per_episode_unintended_summary.csv"),
            "per_sequence_unintended_summary": str(out_dir / "per_sequence_unintended_summary.csv"),
        },
    }
    (out_dir / "diagnostic_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["outputs"], indent=2))


if __name__ == "__main__":
    main()
