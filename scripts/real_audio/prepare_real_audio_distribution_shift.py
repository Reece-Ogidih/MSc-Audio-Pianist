#!/usr/bin/env python3
"""Prepare real-piano audio clips for distribution-shift evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ala_pianist.audio.real_piano import (
    construct_real_audio_benchmarks,
    preprocess_recordings,
    validate_raw_recordings,
)
from ala_pianist.evaluation.final_experiments import write_csv, write_json


ROOT = Path("/home/reece_dev/msc-audio-pianist")
DEFAULT_MANIFESTS = (
    ROOT / "configs/complete_pairwise_v1.json",
    ROOT / "configs/long_horizon_clean_test_v1.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/real_piano/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/real_audio_distribution_shift/prepared_v1")
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows, validation_summary = validate_raw_recordings(args.raw_dir)
    write_csv(args.output_dir / "recording_manifest.csv", validation_rows)
    write_json(args.output_dir / "recording_manifest_summary.json", validation_summary)
    if not validation_summary["ready"]:
        print(json.dumps(validation_summary, indent=2, sort_keys=True))
        raise SystemExit("Real-piano raw recordings are not ready for preprocessing.")
    if args.validate_only:
        print("REAL_AUDIO_RECORDING_VALIDATION_COMPLETE=true")
        return

    processed_dir = args.output_dir / "processed_notes"
    preprocessing_rows, preprocessing_summary = preprocess_recordings(args.raw_dir, processed_dir)
    write_csv(args.output_dir / "preprocessing_manifest.csv", preprocessing_rows)
    write_json(args.output_dir / "preprocessing_manifest_summary.json", preprocessing_summary)

    benchmark_rows, benchmark_summary = construct_real_audio_benchmarks(
        processed_dir=processed_dir,
        output_dir=args.output_dir / "benchmark_audio",
        manifests=DEFAULT_MANIFESTS,
        realizations=args.realizations,
        seed=args.seed,
    )
    # Keep primary benchmark bounded to exact-13 and complete-pairwise rows from
    # complete_pairwise plus clean unseen compositions; downstream evaluator can
    # select exact-13 by sequence names.
    write_csv(args.output_dir / "benchmark_realization_manifest.csv", benchmark_rows)
    write_json(args.output_dir / "benchmark_realization_manifest_summary.json", benchmark_summary)
    print(f"output_dir={args.output_dir}")
    print(f"constructed_clip_count={benchmark_summary['constructed_clip_count']}")
    print("REAL_AUDIO_PREPARATION_COMPLETE=true")


if __name__ == "__main__":
    main()
