#!/usr/bin/env python3
"""Render deterministic Pipeline 2 direct-audio rollouts and motion metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from ala_pianist.evaluation.direct_audio import (
    build_clip_selection,
    build_pipeline2_evaluation_env,
    sequence_name,
)
from ala_pianist.evaluation.final_experiments import write_csv, write_json
from ala_pianist.evaluation.metrics import binary_key_vector, pressed_key_metrics, timestep_key_metrics
from ala_pianist.evaluation.motion_quality import (
    action_quality,
    fingertip_motion_quality,
    per_dimension_action_quality,
)
from ala_pianist.rl import DirectDroQAgent


DEFAULT_MONTAGE = ((72,), (76,), (72, 73), (73, 72), (74, 75), (76, 75))


def parse_sequences(raw: str | None) -> tuple[tuple[int, ...], ...]:
    if not raw:
        return DEFAULT_MONTAGE
    return tuple(tuple(int(part) for part in chunk.split(",")) for chunk in raw.split(";") if chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", default=None)
    parser.add_argument("--evaluation-audio-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=4041)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    sequences = parse_sequences(args.sequences)
    audio_root = args.evaluation_audio_root or args.output_dir / "canonical_rollout_audio"
    env = build_pipeline2_evaluation_env(
        generated_root=audio_root,
        sequences=sequences,
        variants_per_sequence=1,
        split="canonical_eval",
        seed=args.seed,
        device_horizon_steps=96,
    )
    selection = build_clip_selection(env)
    agent = DirectDroQAgent.load(args.checkpoint, device=args.device)
    rollout_rows = []
    per_dim_rows = []
    montage_frames = []
    for index, sequence in enumerate(sequences):
        seq_name = sequence_name(sequence)
        clip_index = selection.sequence_to_clip_index[tuple(sequence)]
        sequence_dir = args.output_dir / seq_name
        sequence_dir.mkdir(parents=True, exist_ok=True)
        metrics, trace_rows, frames = rollout_once(
            env=env,
            agent=agent,
            sequence=tuple(sequence),
            clip_index=clip_index,
            seed=args.seed + index,
            width=args.width,
            height=args.height,
        )
        metrics.update(
            {
                "sequence_name": seq_name,
                "sequence": "-".join(str(pitch) for pitch in sequence),
                "checkpoint_path": str(args.checkpoint),
                "video_path": str(sequence_dir / "rollout.mp4"),
                "trace_path": str(sequence_dir / "trace.csv"),
            }
        )
        write_trace(sequence_dir / "trace.csv", trace_rows)
        write_json(sequence_dir / "summary.json", metrics)
        imageio.mimsave(sequence_dir / "rollout.mp4", frames, fps=args.fps, macro_block_size=1)
        montage_frames.extend(frames)
        rollout_rows.append(metrics)
        trace_df = pd.DataFrame(trace_rows)
        action_cols = sorted(col for col in trace_df.columns if col.startswith("action_"))
        actions = trace_df[action_cols].to_numpy(dtype=float) if action_cols else np.zeros((0, 22))
        for dim in per_dimension_action_quality(actions):
            per_dim_rows.append({"sequence_name": seq_name, **dim})
    if montage_frames:
        imageio.mimsave(args.output_dir / "pipeline2_montage.mp4", montage_frames, fps=args.fps, macro_block_size=1)
    write_csv(args.output_dir / "motion_summary.csv", rollout_rows)
    write_csv(args.output_dir / "per_action_dimension_motion.csv", per_dim_rows)
    write_json(
        args.output_dir / "manifest.json",
        {
            "checkpoint": str(args.checkpoint),
            "sequences": [list(seq) for seq in sequences],
            "video": str(args.output_dir / "pipeline2_montage.mp4"),
            "camera": "MuJoCo camera_id=0, matching previous local Pipeline 1 smoke renderer where possible",
            "audio_mode": "correct",
            "policy_observation_fields": ["audio", "physical"],
        },
    )
    print(f"rollouts={len(rollout_rows)}")
    print(f"montage={args.output_dir / 'pipeline2_montage.mp4'}")
    print(f"motion_summary={args.output_dir / 'motion_summary.csv'}")
    print("PIPELINE2_ROLLOUT_RENDER_COMPLETE=true")


def rollout_once(
    *,
    env,
    agent: DirectDroQAgent,
    sequence: tuple[int, ...],
    clip_index: int,
    seed: int,
    width: int,
    height: int,
) -> tuple[dict, list[dict], list[np.ndarray]]:
    obs, info = env.reset_to_clip_index(clip_index, seed=seed)
    base = env._base_env_for_clip_index(env._active_clip_index)
    trace_rows = []
    frames = []
    target_vectors = []
    pressed_vectors = []
    pressed_seen: set[int] = set()
    max_target = 0.0
    max_unintended = 0.0
    integrated_unintended = 0.0
    actions = []
    dt = env.control_timestep_seconds
    for step in range(env.horizon_steps):
        action = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        states = base.piano_key_states()
        fingertips = base.fingertip_positions()
        row = trace_row(step, step * dt, action, states, fingertips, info, reward)
        trace_rows.append(row)
        frames.append(base.env.physics.render(height=height, width=width, camera_id=0))
        actions.append(np.asarray(action, dtype=np.float32))
        target_vectors.append(binary_key_vector(info["target_keys"]))
        pressed_vectors.append(binary_key_vector(info["pressed_keys"]))
        pressed_seen.update(int(key) for key in info["pressed_keys"])
        max_target = max(max_target, float(info["target_key_state"]))
        step_unintended = float(info["max_unintended_key_state"])
        max_unintended = max(max_unintended, step_unintended)
        integrated_unintended += step_unintended * dt
        if terminated or truncated:
            break
    target_keys = {pitch - 21 for pitch in sequence}
    pressed = pressed_key_metrics(target_keys, pressed_seen)
    timestep = timestep_key_metrics(target_vectors, pressed_vectors)
    trace_df = pd.DataFrame(trace_rows)
    motion = {
        **action_quality(np.asarray(actions)),
        **fingertip_motion_quality(trace_df, dt=dt),
    }
    metrics = {
        "target_keys": sorted(target_keys),
        "pressed_keys": sorted(pressed_seen),
        "pressed_key_precision": pressed.precision,
        "pressed_key_recall": pressed.recall,
        "pressed_key_f1": pressed.f1,
        "timestep_f1": timestep.f1,
        "max_target_key_state": max_target,
        "max_unintended_key_state": max_unintended,
        "integrated_unintended_key_state": integrated_unintended,
        "step_count": len(trace_rows),
        "control_timestep": dt,
        **motion,
    }
    return metrics, trace_rows, frames


def trace_row(step: int, time_seconds: float, action, states, fingertips, info, reward: float) -> dict:
    row = {
        "step_index": int(step),
        "simulation_time": float(time_seconds),
        "target_keys": json.dumps(list(map(int, info.get("target_keys", ())))),
        "pressed_keys": json.dumps(list(map(int, info.get("pressed_keys", ())))),
        "target_key_state": float(info.get("target_key_state", 0.0)),
        "max_unintended_key_state": float(info.get("max_unintended_key_state", 0.0)),
        "reward": float(reward),
        "native_reward": float(info.get("native_reward", 0.0)),
        "shaped_reward": float(info.get("shaped_reward", 0.0)),
    }
    for idx, value in enumerate(np.asarray(action, dtype=float).reshape(-1)):
        row[f"action_{idx:02d}"] = float(value)
    for midi in range(72, 77):
        row[f"key_state_midi_{midi}"] = float(states[midi - 21])
    for name, pos in fingertips.items():
        safe = str(name).replace("/", "_").replace(" ", "_")
        row[f"fingertip_{safe}_x"] = float(pos[0])
        row[f"fingertip_{safe}_y"] = float(pos[1])
        row[f"fingertip_{safe}_z"] = float(pos[2])
    return row


def write_trace(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
