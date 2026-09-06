#!/usr/bin/env python3
"""Evaluate Pipeline 1/2 on prepared real-piano benchmark clips."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from ala_pianist.audio import AudioReferenceBank
from ala_pianist.evaluation import (
    horizon_steps_for_notes,
    load_long_horizon_benchmark,
    sequence_notes,
)
from ala_pianist.evaluation.direct_audio import sha256_file
from ala_pianist.evaluation.final_experiments import write_csv
from ala_pianist.pipelines.indirect import IndirectPipelineConfig, RenderedBenchmarkItem
from ala_pianist.rl import DirectAudioClip, DirectAudioGoalEnv, DirectDroQAgent


ROOT = Path("/home/reece_dev/msc-audio-pianist")
DEFAULT_PREPARED_MANIFEST = ROOT / "artifacts/real_audio_distribution_shift/prepared_v1/benchmark_realization_manifest.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/real_audio_distribution_shift/evaluation_v1"
DEFAULT_PIPELINE1 = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1/checkpoint_800000_steps.pt"
DEFAULT_PIPELINE2 = (
    ROOT
    / "artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1"
    / "lightweight_checkpoints/checkpoint_1000000_steps.pt"
)
EXACT_13_NAMES = {
    "anchor_000_72",
    "anchor_001_73",
    "anchor_002_74",
    "anchor_003_75",
    "anchor_004_76",
    "originally_seen_adjacent_006_72_73",
    "originally_seen_adjacent_010_73_72",
    "originally_seen_adjacent_012_73_74",
    "originally_seen_adjacent_016_74_73",
    "originally_seen_adjacent_018_74_75",
    "originally_seen_adjacent_022_75_74",
    "originally_seen_adjacent_024_75_76",
    "originally_seen_adjacent_028_76_75",
}


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", type=Path, default=DEFAULT_PREPARED_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=["exact_13_retention", "complete_pairwise", "clean_unseen_compositions"],
        default=None,
    )
    parser.add_argument("--realization", type=int, action="append", default=None)
    parser.add_argument("--pipeline1-controller", type=Path, default=DEFAULT_PIPELINE1)
    parser.add_argument("--pipeline2-checkpoint", type=Path, default=DEFAULT_PIPELINE2)
    parser.add_argument("--pipeline2-label", default="Pipeline2 seed13 1M real-audio")
    parser.add_argument("--skip-pipeline1", action="store_true")
    parser.add_argument("--skip-pipeline2", action="store_true")
    parser.add_argument("--pipeline1-condition", choices=["both", "oracle", "transcribed"], default="transcribed")
    parser.add_argument("--include-audio-interventions", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-clips", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evalmod = _load_long_horizon_script()
    items, manifest_rows = _load_real_items(
        args.prepared_manifest,
        benchmarks=tuple(args.benchmark or ("exact_13_retention", "complete_pairwise", "clean_unseen_compositions")),
        realizations=None if args.realization is None else tuple(int(value) for value in args.realization),
        max_clips=args.max_clips,
    )
    if not items:
        raise ValueError("No real-audio clips selected.")
    horizon_steps = max(horizon_steps_for_notes(item.notes) for item in items)

    sequence_rows: list[dict[str, Any]] = []
    transcription_rows: list[dict[str, Any]] = []
    audio_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    if not args.skip_pipeline1:
        p1_rows, p1_transcription, p1_events = evalmod._evaluate_pipeline1(
            items=tuple(items),
            output_dir=args.output_dir / "pipeline1",
            controller_checkpoint=args.pipeline1_controller,
            transcriber_name="basic_pitch",
            condition=args.pipeline1_condition,
            config=IndirectPipelineConfig(
                midi_min=72,
                midi_max=76,
                confidence_threshold=0.3,
                range_policy="drop",
                duplicate_policy="merge",
            ),
            horizon_steps=horizon_steps,
            seed=args.seed,
            device=args.device,
            onset_tolerance=0.05,
            offset_tolerance=0.10,
        )
        sequence_rows.extend(_tag_rows(p1_rows, manifest_rows))
        transcription_rows.extend(_tag_rows(p1_transcription, manifest_rows))
        event_rows.extend(p1_events)

    if not args.skip_pipeline2:
        p2_rows, p2_audio_rows, p2_events = _evaluate_real_pipeline2(
            evalmod=evalmod,
            items=tuple(items),
            output_dir=args.output_dir / "pipeline2",
            checkpoint_path=args.pipeline2_checkpoint,
            model_label=args.pipeline2_label,
            horizon_steps=horizon_steps,
            include_audio_interventions=args.include_audio_interventions,
            device=args.device,
            seed=args.seed,
        )
        sequence_rows.extend(_tag_rows(p2_rows, manifest_rows))
        audio_rows.extend(p2_audio_rows)
        event_rows.extend(p2_events)

    write_csv(args.output_dir / "long_horizon_sequence_metrics.csv", evalmod._json_safe_rows(sequence_rows))
    write_csv(args.output_dir / "long_horizon_event_metrics.csv", evalmod._json_safe_rows(event_rows))
    write_csv(args.output_dir / "pipeline1_transcription_long_horizon.csv", evalmod._json_safe_rows(transcription_rows))
    write_csv(args.output_dir / "audio_dependence_long_horizon.csv", evalmod._json_safe_rows(audio_rows))
    summary = {
        "prepared_manifest": str(args.prepared_manifest),
        "clip_count": len(items),
        "runtime_seconds": time.perf_counter() - started,
        "pipeline1_controller": "" if args.skip_pipeline1 else str(args.pipeline1_controller),
        "pipeline2_checkpoint": "" if args.skip_pipeline2 else str(args.pipeline2_checkpoint),
        "outputs": {
            "long_horizon_sequence_metrics": str(args.output_dir / "long_horizon_sequence_metrics.csv"),
            "pipeline1_transcription_long_horizon": str(args.output_dir / "pipeline1_transcription_long_horizon.csv"),
            "audio_dependence_long_horizon": str(args.output_dir / "audio_dependence_long_horizon.csv"),
        },
    }
    (args.output_dir / "real_audio_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"output_dir={args.output_dir}")
    print(f"clip_count={len(items)}")
    print(f"runtime_seconds={summary['runtime_seconds']:.2f}")
    print("REAL_AUDIO_EVALUATION_COMPLETE=true")


def _load_long_horizon_script():
    script_path = ROOT / "scripts/evaluate_long_horizon_compositional.py"
    spec = importlib.util.spec_from_file_location("evaluate_long_horizon_compositional", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_real_items(
    manifest: Path,
    *,
    benchmarks: tuple[str, ...],
    realizations: tuple[int, ...] | None,
    max_clips: int | None,
) -> tuple[list[RenderedBenchmarkItem], list[dict[str, Any]]]:
    frame = pd.read_csv(manifest)
    selected = _filter_manifest(frame, benchmarks=benchmarks, realizations=realizations)
    if max_clips is not None:
        selected = selected.head(int(max_clips))
    notes_by_key = _notes_by_manifest_sequence(selected)
    items: list[RenderedBenchmarkItem] = []
    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        key = (str(row["manifest_path"]), str(row["sequence_name"]))
        pitches = tuple(int(value) for value in str(row["sequence"]).split("-") if value)
        items.append(
            RenderedBenchmarkItem(
                sequence_name=f"{row['sequence_name']}_realization_{int(row['realization']):02d}",
                pitches=pitches,
                midi_path=Path(row["midi_path"]),
                wav_path=Path(row["wav_path"]),
                notes=notes_by_key[key],
            )
        )
        rows.append(row.to_dict())
    return items, rows


def _filter_manifest(frame: pd.DataFrame, *, benchmarks: tuple[str, ...], realizations: tuple[int, ...] | None) -> pd.DataFrame:
    masks = []
    for benchmark in benchmarks:
        if benchmark == "exact_13_retention":
            masks.append(frame["sequence_name"].isin(EXACT_13_NAMES))
        elif benchmark == "complete_pairwise":
            masks.append(frame["benchmark"].eq("complete_pairwise_v1"))
        elif benchmark == "clean_unseen_compositions":
            masks.append(frame["benchmark"].eq("long_horizon_clean_test_v1"))
        else:
            raise ValueError(f"Unsupported benchmark selection: {benchmark!r}")
    mask = masks[0]
    for extra in masks[1:]:
        mask = mask | extra
    selected = frame[mask].copy()
    if realizations is not None:
        selected = selected[selected["realization"].isin(realizations)]
    return selected.sort_values(["benchmark", "sequence_name", "realization"]).reset_index(drop=True)


def _notes_by_manifest_sequence(frame: pd.DataFrame) -> dict[tuple[str, str], tuple]:
    out = {}
    for manifest_path in sorted({str(value) for value in frame["manifest_path"]}):
        benchmark = load_long_horizon_benchmark(manifest_path, allow_trained_short=True)
        for sequence in benchmark.sequences:
            out[(manifest_path, sequence.name)] = sequence_notes(sequence, benchmark.timing)
    return out


def _evaluate_real_pipeline2(
    *,
    evalmod,
    items: tuple[RenderedBenchmarkItem, ...],
    output_dir: Path,
    checkpoint_path: Path,
    model_label: str,
    horizon_steps: int,
    include_audio_interventions: bool,
    device: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bank, clips = _real_audio_bank_and_clips(items)
    sequences = tuple(dict.fromkeys(tuple(item.pitches) for item in items))
    env = DirectAudioGoalEnv(
        audio_bank=bank,
        clips=tuple(clips),
        sequences=sequences,
        sequence_sampling_weights=tuple(1.0 / len(sequences) for _ in sequences),
        horizon_steps=horizon_steps,
        lookahead=1,
        seed=seed,
        sampling_split="real_audio_eval",
    )
    agent = DirectDroQAgent.load(checkpoint_path, device=device)
    checkpoint_hash = sha256_file(checkpoint_path)
    modes = ("correct", "zero", "mismatched") if include_audio_interventions else ("correct",)
    rows: list[dict[str, Any]] = []
    audio_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for clip_index, item in enumerate(items):
        mismatch_clip_id = clips[(clip_index + 1) % len(clips)].clip_id
        if include_audio_interventions:
            audio_rows.append(
                evalmod._action_dependence_row(
                    agent,
                    env,
                    tuple(item.pitches),
                    clip_index,
                    tuple(clips[(clip_index + 1) % len(clips)].sequence),
                    mismatch_clip_id,
                    model_label,
                )
            )
        for mode in modes:
            row, events = evalmod._rollout_direct_audio(
                agent=agent,
                env=env,
                sequence=tuple(item.pitches),
                notes=item.notes,
                sequence_name=item.sequence_name,
                clip_index=clip_index,
                audio_mode=mode,
                mismatch_sequence=tuple(clips[(clip_index + 1) % len(clips)].sequence),
                mismatch_clip_id=mismatch_clip_id,
                model_label=model_label,
                checkpoint_path=checkpoint_path,
                checkpoint_hash=checkpoint_hash,
                seed=seed + clip_index,
            )
            rows.append(row)
            event_rows.extend(events)
    return rows, audio_rows, event_rows


def _real_audio_bank_and_clips(items: tuple[RenderedBenchmarkItem, ...]) -> tuple[AudioReferenceBank, list[DirectAudioClip]]:
    bank = AudioReferenceBank(sample_rate=16_000, past_context_seconds=0.10, future_context_seconds=0.40)
    clips: list[DirectAudioClip] = []
    for index, item in enumerate(items):
        waveform, sample_rate = sf.read(item.wav_path, always_2d=False)
        if np.asarray(waveform).ndim == 2:
            waveform = np.asarray(waveform, dtype=np.float32).mean(axis=1)
        clip_id = bank.add_waveform(
            waveform,
            name=item.sequence_name,
            source_sample_rate=int(sample_rate),
            metadata={"wav_path": str(item.wav_path), "midi_path": str(item.midi_path), "sequence": item.pitches},
        )
        clips.append(
            DirectAudioClip(
                sequence=tuple(item.pitches),
                midi_path=item.midi_path,
                wav_path=item.wav_path,
                clip_id=clip_id,
                variant_index=index,
                velocity=90,
                gain=1.0,
                split="real_audio_eval",
            )
        )
    return bank, clips


def _tag_rows(rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_stem = {
        f"{row['sequence_name']}_realization_{int(row['realization']):02d}": row
        for row in manifest_rows
    }
    tagged = []
    for row in rows:
        metadata = by_stem.get(str(row.get("sequence_name")), {})
        tagged.append(
            {
                **row,
                "real_audio_benchmark": metadata.get("benchmark", ""),
                "realization": metadata.get("realization", ""),
                "source_wav_path": metadata.get("wav_path", ""),
                "source_wav_sha256": metadata.get("wav_sha256", ""),
            }
        )
    return tagged


if __name__ == "__main__":
    main()
