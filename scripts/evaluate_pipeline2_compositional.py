#!/usr/bin/env python3
"""Zero-shot compositional evaluation for Pipeline 2 direct audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ala_pianist.evaluation.direct_audio import (
    aggregate_checkpoint_rows,
    build_clip_selection,
    build_pipeline2_evaluation_env,
    evaluate_checkpoint,
    write_evaluation_outputs,
)
from ala_pianist.evaluation.final_experiments import (
    COMPOSITIONAL_BENCHMARK_SEQUENCES,
    COMPOSITIONAL_CATEGORIES,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-audio-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3031)
    parser.add_argument("--include-audio-interventions", action="store_true")
    args = parser.parse_args()

    sequences = tuple(COMPOSITIONAL_BENCHMARK_SEQUENCES.values())
    audio_root = args.evaluation_audio_root or args.output_dir / "canonical_composition_audio"
    env = build_pipeline2_evaluation_env(
        generated_root=audio_root,
        sequences=sequences,
        seed=args.seed,
        variants_per_sequence=1,
        split="composition_eval",
    )
    selection = build_clip_selection(env)
    audio_modes = ("correct", "zero", "mismatched") if args.include_audio_interventions else ("correct",)
    sequence_rows, action_rows = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        env=env,
        selection=selection,
        seed=args.seed,
        device=args.device,
        audio_modes=audio_modes,
    )
    name_to_category = {
        "-".join(str(pitch) for pitch in pitches): COMPOSITIONAL_CATEGORIES[name]
        for name, pitches in COMPOSITIONAL_BENCHMARK_SEQUENCES.items()
    }
    for row in sequence_rows:
        row["composition_category"] = name_to_category[row["sequence"]]
        row["sequence_length"] = len(row["sequence"].split("-"))
    checkpoint_rows = aggregate_checkpoint_rows(sequence_rows)
    category_rows = []
    for mode in sorted({row["audio_mode"] for row in sequence_rows}):
        for category in sorted(set(COMPOSITIONAL_CATEGORIES.values())):
            subset = [row for row in sequence_rows if row["audio_mode"] == mode and row["composition_category"] == category]
            if not subset:
                continue
            category_rows.append(
                {
                    "audio_mode": mode,
                    "composition_category": category,
                    "sequence_count": len(subset),
                    "pressed_key_f1_mean": sum(float(row["pressed_key_f1"]) for row in subset) / len(subset),
                    "timestep_f1_mean": sum(float(row["timestep_f1"]) for row in subset) / len(subset),
                    "max_unintended_mean": sum(float(row["max_unintended_key_state"]) for row in subset) / len(subset),
                    "wrong_press_count_total": sum(int(row["wrong_press_count"]) for row in subset),
                }
            )
    manifest = {
        "benchmark": "pipeline2_compositional_zero_shot_v1",
        "sequences": {name: list(pitches) for name, pitches in COMPOSITIONAL_BENCHMARK_SEQUENCES.items()},
        "categories": COMPOSITIONAL_CATEGORIES,
        "primary_condition": "correct audio",
        "audio_interventions_included": bool(args.include_audio_interventions),
    }
    summary = write_evaluation_outputs(
        output_dir=args.output_dir,
        checkpoint_rows=checkpoint_rows,
        sequence_rows=sequence_rows,
        action_rows=action_rows,
        manifest=manifest,
    )
    write_csv(args.output_dir / "composition_category_metrics.csv", category_rows)
    (args.output_dir / "composition_sequences.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    best = summary.get("best_checkpoint_by_correct_pressed_key_f1", {})
    print(f"sequence_count={len(sequences)}")
    print(f"best_correct_f1={float(best.get('pressed_key_f1_mean', 0.0)):.3f}")
    print(f"output_dir={args.output_dir}")
    print("COMPOSITIONAL_EVALUATION_COMPLETE=true")


if __name__ == "__main__":
    main()
