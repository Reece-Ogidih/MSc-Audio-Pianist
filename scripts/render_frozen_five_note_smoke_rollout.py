#!/usr/bin/env python
"""Render a synchronized smoke rollout for the frozen five-note controller."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path("/home/reece_dev/msc-audio-pianist")
sys.path.insert(0, str(ROOT / "scripts"))

from ala_pianist.evaluation.unintended import classify_unintended_keys  # noqa: E402
from ala_pianist.evaluation.motion_quality import annotate_rollout_frame, phase_labels  # noqa: E402
from ala_pianist.music import assign_right_hand_fingering, sequence_timing_from_profile, write_sequence_midi  # noqa: E402
from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv  # noqa: E402
from evaluate_general_one_hand_policy import reward_config_from_profile  # noqa: E402


DEFAULT_RELEASE_DIR = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1"
DEFAULT_SEQUENCE_NAME = "trained_73_72"
DEFAULT_PITCHES = (73, 72)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def control_timestep(env: GeneralOneHandGoalEnv) -> float:
    for candidate in [
        getattr(env.env, "control_timestep", None),
        getattr(env.task, "control_timestep", None),
    ]:
        if callable(candidate):
            value = float(candidate())
            if value > 0:
                return value
        elif candidate is not None:
            value = float(candidate)
            if value > 0:
                return value
    return float(env.env.physics.model.opt.timestep)


def target_mask(info: dict[str, Any]) -> list[int]:
    mask = [0] * 88
    for key in info.get("target_keys", ()):
        mask[int(key)] = 1
    return mask


def trace_row(
    *,
    step_index: int,
    frame_index: int,
    simulation_time: float,
    info: dict[str, Any],
    action: np.ndarray,
    states: np.ndarray,
    fingertips: dict[str, np.ndarray],
    reward: float,
) -> dict[str, Any]:
    target_keys = tuple(int(k) for k in info.get("target_keys", ()))
    classifications = classify_unintended_keys(
        states,
        current_target_keys=target_keys,
        previous_target_keys=info.get("previous_target_keys", ()),
        future_target_keys=info.get("future_target_keys", ()),
        press_threshold=0.5,
    )
    wrong_events = [
        {"key_index": item.key_index, "category": item.category, "value": item.value}
        for item in classifications
        if item.is_pressed
    ]
    components = info.get("reward_components", {})
    row: dict[str, Any] = {
        "step_index": step_index,
        "frame_index": frame_index,
        "simulation_time": simulation_time,
        "intended_midi_pitches": json.dumps([key + 21 for key in target_keys]),
        "intended_key_indices": json.dumps(list(target_keys)),
        "intended_key_mask": json.dumps(target_mask(info)),
        "pressed_keys": json.dumps(list(map(int, info.get("pressed_keys", ())))),
        "pressed_key_indices": json.dumps(list(map(int, info.get("pressed_keys", ())))),
        "pressed_midi_pitches": json.dumps([int(key) + 21 for key in info.get("pressed_keys", ())]),
        "target_key_activation": float(info.get("target_key_state", 0.0)),
        "max_unintended_key_state": float(info.get("max_unintended_key_state", 0.0)),
        "wrong_key_threshold_crossing_events": json.dumps(wrong_events),
        "previous_target_release_state": float(components.get("release_previous_key_state", 0.0)),
        "onset_event": int(bool(target_keys)),
        "release_event": int(not target_keys and bool(info.get("previous_target_keys", ()))),
        "reward": float(reward),
        "shaped_reward": float(info.get("shaped_reward", 0.0)),
        "native_reward": float(info.get("native_reward", 0.0)),
    }
    for midi in range(72, 77):
        row[f"key_state_midi_{midi}"] = float(states[midi - 21])
    for idx, value in enumerate(action):
        row[f"action_{idx:02d}"] = float(value)
    for name, pos in fingertips.items():
        safe = name.replace("/", "_").replace(" ", "_")
        row[f"fingertip_{safe}_x"] = float(pos[0])
        row[f"fingertip_{safe}_y"] = float(pos[1])
        row[f"fingertip_{safe}_z"] = float(pos[2])
    for key, value in components.items():
        row[f"reward_component_{key}"] = float(value)
    return row


def write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    width = 1100
    height = 580
    margin_l = 70
    margin_r = 30
    margin_t = 30
    row_h = 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    series = [
        ("target", "target_key_activation", (20, 90, 180)),
        ("unintended", "max_unintended_key_state", (190, 60, 40)),
        ("midi72", "key_state_midi_72", (80, 130, 60)),
        ("midi73", "key_state_midi_73", (120, 60, 150)),
    ]
    n = max(1, len(rows) - 1)
    for sidx, (label, column, color) in enumerate(series):
        top = margin_t + sidx * row_h
        bottom = top + row_h - 25
        draw.text((10, top + 25), label, fill=color)
        draw.line((margin_l, bottom, width - margin_r, bottom), fill=(180, 180, 180))
        draw.line((margin_l, top, margin_l, bottom), fill=(180, 180, 180))
        points = []
        for idx, row in enumerate(rows):
            x = margin_l + (width - margin_l - margin_r) * idx / n
            y = bottom - (bottom - top) * max(0.0, min(1.0, float(row.get(column, 0.0))))
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        for idx, row in enumerate(rows):
            if int(row.get("onset_event", 0)):
                x = margin_l + (width - margin_l - margin_r) * idx / n
                draw.line((x, top, x, bottom), fill=(220, 220, 120))
            if int(row.get("release_event", 0)):
                x = margin_l + (width - margin_l - margin_r) * idx / n
                draw.line((x, top, x, bottom), fill=(170, 170, 170))
    draw.text((margin_l, height - 50), "x-axis: control steps / simulation time; yellow=target frame, grey=release frame", fill=(40, 40, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def render_rollout(
    *,
    release_dir: Path,
    sequence_name: str = DEFAULT_SEQUENCE_NAME,
    pitches: tuple[int, ...] = DEFAULT_PITCHES,
    width: int = 640,
    height: int = 480,
    annotate: bool = False,
    output_subdir: str | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    manifest = load_json(release_dir / "manifest.json")
    config = load_json(release_dir / "provenance/resolved_training_config.json")
    checkpoint = release_dir / manifest["frozen_checkpoint_relative_path"]
    out_dir = release_dir / "validation/smoke_rollout" / (output_subdir or sequence_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_path = write_sequence_midi(
        list(pitches),
        out_dir / f"{sequence_name}_{'_'.join(map(str, pitches))}.mid",
        midi_min=min(pitches),
        midi_max=max(pitches),
        timing=sequence_timing_from_profile(config["sequence_timing_profile"]),
        fingering_fn=assign_right_hand_fingering,
        title=f"frozen smoke {sequence_name}",
    )
    env = GeneralOneHandGoalEnv(
        midi_path=midi_path,
        midi_min=min(pitches),
        midi_max=max(pitches),
        seed=config["seed"],
        lookahead=config["lookahead"],
        horizon_steps=128,
        action_mode=config["action_mode"],
        action_repeat=config["action_repeat"],
        reward_config=reward_config_from_profile(config["reward_profile"]),
    )
    policy = DroQPolicy.load(checkpoint, device="cpu")
    dt = control_timestep(env)
    fps = max(1, int(round(1.0 / dt)))
    obs, info = env.reset(seed=config["seed"])
    rows: list[dict[str, Any]] = []
    frames = []
    terminated = False
    truncated = False
    step = 0
    render_failed: str | None = None
    while not (terminated or truncated) and step < env.horizon_steps:
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        states = env.piano_key_states()
        row = trace_row(
            step_index=step,
            frame_index=len(frames),
            simulation_time=step * dt,
            info=info,
            action=action,
            states=states,
            fingertips=env.fingertip_positions(),
            reward=reward,
        )
        rows.append(row)
        try:
            frame = env.env.physics.render(height=height, width=width, camera_id=0)
            if annotate:
                phase = str(phase_labels(pd.DataFrame(rows)).iloc[-1])
                intended_midi = json.loads(row["intended_midi_pitches"])
                intended_keys = json.loads(row["intended_key_indices"])
                pressed_midi = json.loads(row["pressed_midi_pitches"])
                target_keys = {int(key) for key in intended_keys}
                wrong_pressed = [midi for midi in pressed_midi if int(midi) - 21 not in target_keys]
                frame = annotate_rollout_frame(
                    frame,
                    sequence_id=sequence_name,
                    simulation_time=float(row["simulation_time"]),
                    intended_midi=intended_midi,
                    intended_keys=intended_keys,
                    pressed_midi=pressed_midi,
                    wrong_pressed_midi=wrong_pressed,
                    phase=phase,
                    failure_labels=[],
                    target_activation=float(row["target_key_activation"]),
                    max_unintended=float(row["max_unintended_key_state"]),
                )
            frames.append(frame)
        except Exception as exc:
            render_failed = f"{type(exc).__name__}: {exc}"
            break
        step += 1
    if render_failed:
        summary = {
            "render_succeeded": False,
            "render_error": render_failed,
            "trace_rows": len(rows),
            "midi_path": str(midi_path),
            "control_timestep": dt,
            "fps": fps,
        }
        write_json(out_dir / "summary.json", summary)
        if rows:
            write_trace(out_dir / "trace.csv", rows)
            write_plot(out_dir / "diagnostic_plot.png", rows)
        raise RuntimeError(render_failed)
    video_path = out_dir / "rollout.mp4"
    imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)
    trace_path = out_dir / "trace.csv"
    plot_path = out_dir / "diagnostic_plot.png"
    write_trace(trace_path, rows)
    write_plot(plot_path, rows)
    pressed = sorted({key for row in rows for key in json.loads(row["pressed_keys"])})
    summary = {
        "render_succeeded": True,
        "sequence_name": sequence_name,
        "pitches": list(pitches),
        "midi_path": str(midi_path),
        "video_path": str(video_path),
        "trace_path": str(trace_path),
        "plot_path": str(plot_path),
        "frame_count": len(frames),
        "trace_rows": len(rows),
        "control_timestep": dt,
        "fps": fps,
        "pressed_keys": pressed,
        "max_target_key_activation": max((row["target_key_activation"] for row in rows), default=0.0),
        "max_unintended_key_state": max((row["max_unintended_key_state"] for row in rows), default=0.0),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def parse_pitches(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split(",") if part)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--sequence-name", default=DEFAULT_SEQUENCE_NAME)
    parser.add_argument("--pitches", default="73,72")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--output-subdir")
    args = parser.parse_args()
    result = render_rollout(
        release_dir=args.release_dir,
        sequence_name=args.sequence_name,
        pitches=parse_pitches(args.pitches),
        width=args.width,
        height=args.height,
        annotate=args.annotate,
        output_subdir=args.output_subdir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
