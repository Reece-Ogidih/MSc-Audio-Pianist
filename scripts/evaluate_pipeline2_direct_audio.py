#!/usr/bin/env python3
"""Evaluate Pipeline 2 direct raw-audio DroQ checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess

from ala_pianist.evaluation.direct_audio import (
    aggregate_checkpoint_rows,
    build_clip_selection,
    build_pipeline2_evaluation_env,
    checkpoint_step,
    discover_lightweight_checkpoints,
    evaluate_checkpoint,
    manifest_for_env,
    parse_sequences,
    write_evaluation_outputs,
    write_pipeline_comparison,
)


ROOT = Path("/home/reece_dev/msc-audio-pianist")


def git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--evaluation-audio-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--sequences", default=None)
    parser.add_argument("--horizon-steps", type=int, default=64)
    parser.add_argument("--pipeline1-summary", type=Path, default=ROOT / "experiments/indirect_pipeline/five_note_rendered_benchmark/end_to_end_summary.json")
    args = parser.parse_args()

    if args.run_dir is None and not args.checkpoint:
        raise ValueError("Provide --run-dir or at least one --checkpoint.")
    checkpoints = list(args.checkpoint)
    if args.run_dir is not None:
        checkpoints.extend(discover_lightweight_checkpoints(args.run_dir))
    checkpoints = sorted(set(Path(path) for path in checkpoints), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError("No checkpoints selected for evaluation.")

    output_dir = args.output_dir
    if output_dir is None:
        if args.run_dir is None:
            output_dir = checkpoints[0].parent / "evaluation"
        else:
            output_dir = args.run_dir / "evaluation"
    audio_root = args.evaluation_audio_root or output_dir / "canonical_eval_audio"
    sequences = parse_sequences(args.sequences)
    env = build_pipeline2_evaluation_env(
        generated_root=audio_root,
        sequences=sequences,
        seed=args.seed,
        device_horizon_steps=args.horizon_steps,
    )
    env.assert_no_forbidden_observation_fields()
    selection = build_clip_selection(env)

    all_sequence_rows = []
    all_action_rows = []
    for checkpoint in checkpoints:
        sequence_rows, action_rows = evaluate_checkpoint(
            checkpoint_path=checkpoint,
            env=env,
            selection=selection,
            seed=args.seed,
            device=args.device,
        )
        all_sequence_rows.extend(sequence_rows)
        all_action_rows.extend(action_rows)

    checkpoint_rows = aggregate_checkpoint_rows(all_sequence_rows)
    manifest = manifest_for_env(env, selection)
    manifest.update(
        {
            "run_dir": str(args.run_dir) if args.run_dir else None,
            "checkpoint_paths": [str(path) for path in checkpoints],
            "output_dir": str(output_dir),
            "device": args.device,
            "seed": args.seed,
            "hostname": socket.gethostname(),
            "git_commit": git_commit(Path.cwd()),
            "determinism_policy": "deterministic actor inference; one rollout per sequence/audio mode because resets are deterministic for fixed clip and seed",
        }
    )
    summary = write_evaluation_outputs(
        output_dir=output_dir,
        checkpoint_rows=checkpoint_rows,
        sequence_rows=all_sequence_rows,
        action_rows=all_action_rows,
        manifest=manifest,
    )
    comparison_path = write_pipeline_comparison(
        output_dir=output_dir,
        checkpoint_rows=checkpoint_rows,
        pipeline1_summary_path=args.pipeline1_summary,
    )
    if comparison_path is not None:
        summary["pipeline_comparison_path"] = str(comparison_path)
        (Path(output_dir) / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    correct_rows = [row for row in checkpoint_rows if row["audio_mode"] == "correct"]
    print("checkpoint | correct_f1 | zero_f1 | mismatch_f1 | correct_timestep_f1")
    by_step = {}
    for row in checkpoint_rows:
        by_step.setdefault(row["checkpoint_step"], {})[row["audio_mode"]] = row
    for step in sorted(by_step):
        correct = by_step[step].get("correct", {})
        zero = by_step[step].get("zero", {})
        mismatch = by_step[step].get("mismatched", {})
        print(
            f"{step} | "
            f"{float(correct.get('pressed_key_f1_mean', 0.0)):.3f} | "
            f"{float(zero.get('pressed_key_f1_mean', 0.0)):.3f} | "
            f"{float(mismatch.get('pressed_key_f1_mean', 0.0)):.3f} | "
            f"{float(correct.get('timestep_f1_mean', 0.0)):.3f}"
        )
    print(f"best_correct_f1_checkpoint={summary.get('best_checkpoint_by_correct_pressed_key_f1', {}).get('checkpoint_step')}")
    if correct_rows:
        best = max(correct_rows, key=lambda row: (float(row["pressed_key_f1_mean"]), int(row["checkpoint_step"])))
        print(
            "audio_dependence_deltas="
            f"f1_zero:{float(best.get('delta_pressed_key_f1_correct_minus_zero', 0.0)):.3f},"
            f"f1_mismatch:{float(best.get('delta_pressed_key_f1_correct_minus_mismatch', 0.0)):.3f},"
            f"ts_zero:{float(best.get('delta_timestep_f1_correct_minus_zero', 0.0)):.3f},"
            f"ts_mismatch:{float(best.get('delta_timestep_f1_correct_minus_mismatch', 0.0)):.3f}"
        )
    print(f"output_dir={output_dir}")
    print("EVALUATION_COMPLETE=true")


if __name__ == "__main__":
    main()
