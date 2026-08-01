"""Compare raw vs minimally cleaned Basic Pitch predictions for Pipeline 1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_indirect_five_note_benchmark import (  # noqa: E402
    DEFAULT_CONTROLLER,
    _build_transcriber,
    _json_safe_row,
    _performance_assessment,
    _render_or_load_benchmark,
    _safe_transcribe,
    _summary_rows,
    _symbolic_result,
    _transcriber_dependency,
    evaluate_controller_sequence,
)

from ala_pianist.audio.transcriber import TranscriptionOutput  # noqa: E402
from ala_pianist.evaluation.transcription_metrics import transcription_note_metrics  # noqa: E402
from ala_pianist.music import (  # noqa: E402
    MonophonicCleanupConfig,
    TimedNote,
    clean_monophonic_timed_notes,
)
from ala_pianist.pipelines.indirect import (  # noqa: E402
    BENCHMARK_SEQUENCE_PITCHES,
    IndirectPipelineConfig,
    find_default_soundfont,
    write_controller_midi_from_result,
    write_csv_rows,
)
from ala_pianist.rl import DroQPolicy  # noqa: E402


ROOT = Path("/home/reece_dev/msc-audio-pianist")


def main() -> None:
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-dir",
        default=str(ROOT / "experiments" / "indirect_pipeline" / "five_note_rendered_benchmark"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            ROOT
            / "experiments"
            / "indirect_pipeline"
            / "five_note_rendered_benchmark"
            / "final_cleanup_comparison"
        ),
    )
    parser.add_argument("--controller-checkpoint", default=str(DEFAULT_CONTROLLER))
    parser.add_argument("--transcriber", default="basic_pitch", choices=["basic_pitch", "generated_wav_peak"])
    parser.add_argument("--midi-min", type=int, default=72)
    parser.add_argument("--midi-max", type=int, default=76)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument("--offset-tolerance", type=float, default=0.10)
    parser.add_argument("--horizon-steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--checkpoint-device", default="cpu")
    parser.add_argument("--basic-pitch-onset-threshold", type=float, default=0.5)
    parser.add_argument("--basic-pitch-frame-threshold", type=float, default=0.3)
    parser.add_argument("--basic-pitch-min-note-ms", type=float, default=80.0)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    soundfont = find_default_soundfont()
    items = _render_or_load_benchmark(
        benchmark_dir,
        render_audio=False,
        soundfont=soundfont,
        sample_rate=44100,
        gain=0.5,
    )
    config = IndirectPipelineConfig(
        midi_min=args.midi_min,
        midi_max=args.midi_max,
        confidence_threshold=args.confidence_threshold,
        range_policy="drop",
        duplicate_policy="merge",
    )
    transcriber = _build_transcriber(args)
    policy = DroQPolicy.load(args.controller_checkpoint, device=args.checkpoint_device)

    cleanup_config = MonophonicCleanupConfig()
    transcription_rows = []
    cleanup_diagnostics = []
    controller_rows = []
    comparison_rows = []
    raw_controller_rows = []
    cleaned_controller_rows = []
    oracle_controller_rows = []
    raw_predictions_payload = {}
    cleaned_predictions_payload = {}

    for item in items:
        print(f"cleanup_sequence={item.sequence_name} pitches={list(item.pitches)}")
        raw_output = _safe_transcribe(transcriber, item.wav_path)
        cleaned_notes, diagnostics = clean_monophonic_timed_notes(
            raw_output.notes,
            config=cleanup_config,
        )
        cleaned_output = TranscriptionOutput(
            notes=cleaned_notes,
            transcriber_name=f"{raw_output.transcriber_name}_monophonic_cleanup",
            source_audio_path=raw_output.source_audio_path,
            metadata={
                **raw_output.metadata,
                "postprocessor": "clean_monophonic_timed_notes",
                "cleanup_config": cleanup_config.__dict__,
                "cleanup_diagnostics": diagnostics.as_dict(),
            },
        )

        oracle_output = _oracle_output(item)
        oracle_symbolic = _symbolic_result(oracle_output, item.notes, config, args)
        raw_symbolic = _symbolic_result(raw_output, item.notes, config, args)
        cleaned_symbolic = _symbolic_result(cleaned_output, item.notes, config, args)

        raw_metrics = _transcription_metric_bundle(item.notes, raw_symbolic.controller_sequence.notes, args)
        cleaned_metrics = _transcription_metric_bundle(
            item.notes,
            cleaned_symbolic.controller_sequence.notes,
            args,
        )
        diagnostic_row = {
            "sequence_name": item.sequence_name,
            **diagnostics.as_dict(),
            **_raw_failure_diagnostics(item.notes, raw_output.notes),
        }
        cleanup_diagnostics.append(_json_safe_row(diagnostic_row))
        transcription_rows.append(
            _json_safe_row(
                {
                    "sequence_name": item.sequence_name,
                    "pitches": "-".join(map(str, item.pitches)),
                    **{f"raw_{key}": value for key, value in raw_metrics.items()},
                    **{f"cleaned_{key}": value for key, value in cleaned_metrics.items()},
                    "delta_cleaned_minus_raw_note_f1": cleaned_metrics["note_f1"] - raw_metrics["note_f1"],
                    "delta_cleaned_minus_raw_offset_mae_seconds": _delta_optional(
                        cleaned_metrics["offset_mae_seconds"],
                        raw_metrics["offset_mae_seconds"],
                    ),
                    "delta_cleaned_minus_raw_duration_mae_seconds": _delta_optional(
                        cleaned_metrics["duration_mae_seconds"],
                        raw_metrics["duration_mae_seconds"],
                    ),
                    "raw_duplicate_count": diagnostics.raw_same_pitch_duplicate_count,
                    "cleaned_duplicate_count": diagnostics.cleaned_same_pitch_duplicate_count,
                    "raw_overlap_count": diagnostics.raw_overlap_count,
                    "cleaned_overlap_count": diagnostics.cleaned_overlap_count,
                }
            )
        )
        raw_predictions_payload[item.sequence_name] = [_note_payload(note) for note in raw_output.notes]
        cleaned_predictions_payload[item.sequence_name] = [_note_payload(note) for note in cleaned_notes]

        oracle_midi = write_controller_midi_from_result(
            oracle_symbolic,
            output_dir / "controller_goals" / "oracle" / f"{item.sequence_name}.mid",
        )
        raw_midi = None
        cleaned_midi = None
        if raw_symbolic.controller_sequence.notes:
            raw_midi = write_controller_midi_from_result(
                raw_symbolic,
                output_dir / "controller_goals" / "raw" / f"{item.sequence_name}.mid",
            )
        if cleaned_symbolic.controller_sequence.notes:
            cleaned_midi = write_controller_midi_from_result(
                cleaned_symbolic,
                output_dir / "controller_goals" / "cleaned" / f"{item.sequence_name}.mid",
            )

        oracle_row = evaluate_controller_sequence(
            controller_midi_path=oracle_midi,
            reference_notes=item.notes,
            policy=policy,
            seed=args.seed,
            horizon_steps=args.horizon_steps,
            midi_min=args.midi_min,
            midi_max=args.midi_max,
            condition="oracle",
            sequence_name=item.sequence_name,
        )
        raw_row = evaluate_controller_sequence(
            controller_midi_path=raw_midi,
            reference_notes=item.notes,
            policy=policy,
            seed=args.seed,
            horizon_steps=args.horizon_steps,
            midi_min=args.midi_min,
            midi_max=args.midi_max,
            condition="raw_basic_pitch",
            sequence_name=item.sequence_name,
        )
        cleaned_row = evaluate_controller_sequence(
            controller_midi_path=cleaned_midi,
            reference_notes=item.notes,
            policy=policy,
            seed=args.seed,
            horizon_steps=args.horizon_steps,
            midi_min=args.midi_min,
            midi_max=args.midi_max,
            condition="cleaned_basic_pitch",
            sequence_name=item.sequence_name,
        )
        oracle_controller_rows.append(oracle_row)
        raw_controller_rows.append(raw_row)
        cleaned_controller_rows.append(cleaned_row)
        controller_rows.extend([oracle_row, raw_row, cleaned_row])
        comparison_rows.append(_raw_cleaned_controller_comparison(raw_row, cleaned_row))

    transcription_summary = _summary_rows(transcription_rows, "raw_vs_cleaned_transcription")
    controller_summary = {
        "oracle": _summary_rows(oracle_controller_rows, "oracle"),
        "raw_basic_pitch": _summary_rows(raw_controller_rows, "raw_basic_pitch"),
        "cleaned_basic_pitch": _summary_rows(cleaned_controller_rows, "cleaned_basic_pitch"),
        "cleaned_minus_raw": _summary_rows(comparison_rows, "cleaned_minus_raw_controller"),
    }
    decision = _select_final_path(transcription_summary, controller_summary)
    runtime = time.perf_counter() - start

    write_csv_rows(transcription_rows, output_dir / "raw_vs_cleaned_transcription_per_sequence.csv")
    write_csv_rows(controller_rows, output_dir / "oracle_raw_cleaned_controller_per_sequence.csv")
    write_csv_rows(comparison_rows, output_dir / "raw_vs_cleaned_controller_delta.csv")
    write_csv_rows(cleanup_diagnostics, output_dir / "cleanup_diagnostics_per_sequence.csv")
    (output_dir / "raw_vs_cleaned_transcription_summary.json").write_text(
        json.dumps(transcription_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "oracle_raw_cleaned_controller_summary.json").write_text(
        json.dumps(controller_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "cleanup_diagnostics.json").write_text(
        json.dumps(
            {
                "per_sequence": cleanup_diagnostics,
                "summary": _summary_rows(cleanup_diagnostics, "cleanup_diagnostics"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "raw_predictions.json").write_text(
        json.dumps(raw_predictions_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "cleaned_predictions.json").write_text(
        json.dumps(cleaned_predictions_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    final_summary = {
        "pipeline_1_architecturally_complete": True,
        "pipeline_1_frozen_for_comparison": True,
        "selected_final_path": decision["selected_final_path"],
        "decision_reasons": decision["decision_reasons"],
        "performance_assessment": _performance_assessment(
            cleaned_controller_rows if decision["selected_final_path"] == "CLEANED" else raw_controller_rows
        ),
        "runtime_seconds": runtime,
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(output_dir),
        "transcription_summary": transcription_summary,
        "controller_summary": controller_summary,
        "cleanup_config": cleanup_config.__dict__,
    }
    (output_dir / "end_to_end_final_summary.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_final_report(output_dir / "pipeline_1_v1_final_report.md", final_summary)
    manifest_path = _write_frozen_manifest(
        final_summary=final_summary,
        controller_checkpoint=Path(args.controller_checkpoint),
        output_dir=output_dir,
        args=args,
    )
    print(f"output_dir={output_dir}")
    print(f"manifest_path={manifest_path}")
    print(f"selected_final_path={decision['selected_final_path']}")
    print("pipeline_1_architecturally_complete=True")
    print("pipeline_1_frozen_for_comparison=True")
    print(f"runtime_seconds={runtime:.2f}")


def _oracle_output(item):
    from ala_pianist.audio.transcriber import OracleMidiTranscriber

    return OracleMidiTranscriber(item.midi_path).transcribe(item.wav_path)


def _transcription_metric_bundle(reference_notes, predicted_notes, args) -> dict:
    primary = transcription_note_metrics(
        reference_notes,
        predicted_notes,
        onset_tolerance_seconds=args.onset_tolerance,
        offset_tolerance_seconds=None,
    )
    with_offset = transcription_note_metrics(
        reference_notes,
        predicted_notes,
        onset_tolerance_seconds=args.onset_tolerance,
        offset_tolerance_seconds=args.offset_tolerance,
    )
    payload = primary.as_dict()
    payload.update(
        {
            "with_offset_note_precision": with_offset.note_precision,
            "with_offset_note_recall": with_offset.note_recall,
            "with_offset_note_f1": with_offset.note_f1,
        }
    )
    return payload


def _raw_failure_diagnostics(reference_notes, raw_notes) -> dict:
    raw_ordered = sorted(tuple(raw_notes), key=lambda note: (note.onset, note.offset, note.pitch))
    reference_ordered = sorted(tuple(reference_notes), key=lambda note: (note.onset, note.offset, note.pitch))
    offsets_beyond_next_predicted = 0
    offsets_beyond_next_reference = 0
    for index, note in enumerate(raw_ordered[:-1]):
        if note.offset > raw_ordered[index + 1].onset:
            offsets_beyond_next_predicted += 1
    for note in raw_ordered:
        next_reference_onsets = [
            ref.onset for ref in reference_ordered if ref.onset > note.onset
        ]
        if next_reference_onsets and note.offset > min(next_reference_onsets):
            offsets_beyond_next_reference += 1
    matched = transcription_note_metrics(
        reference_ordered,
        raw_ordered,
        onset_tolerance_seconds=0.05,
        offset_tolerance_seconds=None,
    )
    duplicate_fp = sum(1 for pitch in matched.false_positive_pitches if pitch in {n.pitch for n in reference_ordered})
    return {
        "raw_offsets_beyond_next_predicted_onset": offsets_beyond_next_predicted,
        "raw_offsets_beyond_next_reference_onset": offsets_beyond_next_reference,
        "raw_false_positive_count": matched.false_positive_count,
        "raw_duplicate_false_positive_count": duplicate_fp,
    }


def _raw_cleaned_controller_comparison(raw_row: dict, cleaned_row: dict) -> dict:
    keys = [
        "pressed_key_precision",
        "pressed_key_recall",
        "pressed_key_f1",
        "timestep_f1",
        "max_unintended_key_state",
        "integrated_unintended_key_state",
        "wrong_key_crossings",
        "transition_completion",
        "second_target_completion",
        "previous_target_late_release_duration",
    ]
    row = {"sequence_name": raw_row["sequence_name"]}
    for key in keys:
        raw_value = raw_row.get(key)
        cleaned_value = cleaned_row.get(key)
        row[f"raw_{key}"] = raw_value
        row[f"cleaned_{key}"] = cleaned_value
        row[f"delta_cleaned_minus_raw_{key}"] = _delta_optional(cleaned_value, raw_value)
    row["raw_pressed_keys"] = raw_row.get("pressed_keys")
    row["cleaned_pressed_keys"] = cleaned_row.get("pressed_keys")
    row["raw_strict_outcome"] = raw_row.get("strict_outcome")
    row["cleaned_strict_outcome"] = cleaned_row.get("strict_outcome")
    return _json_safe_row(row)


def _select_final_path(transcription_summary: dict, controller_summary: dict) -> dict:
    reasons = []
    raw_recall = transcription_summary.get("raw_note_recall_mean", 0.0)
    cleaned_recall = transcription_summary.get("cleaned_note_recall_mean", 0.0)
    raw_onset = transcription_summary.get("raw_onset_mae_seconds_mean")
    cleaned_onset = transcription_summary.get("cleaned_onset_mae_seconds_mean")
    raw_offset = transcription_summary.get("raw_offset_mae_seconds_mean")
    cleaned_offset = transcription_summary.get("cleaned_offset_mae_seconds_mean")
    raw_dupes = transcription_summary.get("raw_duplicate_count_mean", 0.0)
    cleaned_dupes = transcription_summary.get("cleaned_duplicate_count_mean", 0.0)
    raw_ctrl = controller_summary["raw_basic_pitch"]
    cleaned_ctrl = controller_summary["cleaned_basic_pitch"]
    recall_ok = cleaned_recall >= raw_recall - 0.05
    onset_ok = cleaned_onset is None or raw_onset is None or cleaned_onset <= raw_onset + 0.005
    offset_better = (
        cleaned_offset is not None
        and raw_offset is not None
        and cleaned_offset < raw_offset
    )
    duplicates_better = cleaned_dupes < raw_dupes
    controller_comparable = (
        cleaned_ctrl.get("pressed_key_f1_mean", 0.0) >= raw_ctrl.get("pressed_key_f1_mean", 0.0) - 0.02
        and cleaned_ctrl.get("timestep_f1_mean", 0.0) >= raw_ctrl.get("timestep_f1_mean", 0.0) - 0.02
    )
    unintended_ok = cleaned_ctrl.get("max_unintended_key_state_mean", 0.0) <= raw_ctrl.get(
        "max_unintended_key_state_mean",
        0.0,
    ) + 0.02
    if recall_ok:
        reasons.append("cleaned recall did not materially regress")
    if onset_ok:
        reasons.append("cleaned onset accuracy remained effectively unchanged")
    if offset_better:
        reasons.append("cleaned offset/duration quality improved")
    if duplicates_better:
        reasons.append("same-pitch duplicate predictions decreased")
    if controller_comparable:
        reasons.append("controller F1/timestep F1 remained comparable")
    if unintended_ok:
        reasons.append("controller unintended activation did not materially worsen")
    selected = (
        "CLEANED"
        if recall_ok and onset_ok and offset_better and duplicates_better and controller_comparable and unintended_ok
        else "RAW"
    )
    if selected == "RAW":
        reasons.append("cleaned path failed at least one selection gate")
    return {"selected_final_path": selected, "decision_reasons": reasons}


def _write_final_report(path: Path, summary: dict) -> None:
    lines = [
        "# Pipeline 1 v1 Final Cleanup Comparison",
        "",
        f"`pipeline_1_architecturally_complete`: `{str(summary['pipeline_1_architecturally_complete']).lower()}`",
        f"`pipeline_1_frozen_for_comparison`: `{str(summary['pipeline_1_frozen_for_comparison']).lower()}`",
        f"`selected_final_path`: `{summary['selected_final_path']}`",
        "",
        "## Decision Reasons",
        "",
        *[f"- {reason}" for reason in summary["decision_reasons"]],
        "",
        "## Summary JSON",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_frozen_manifest(*, final_summary: dict, controller_checkpoint: Path, output_dir: Path, args) -> Path:
    manifest_dir = ROOT / "artifacts" / "frozen_models" / "indirect_pipeline_five_note_v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    payload = {
        "pipeline_name": "indirect_pipeline_five_note",
        "pipeline_version": "v1",
        "pipeline_1_architecturally_complete": True,
        "pipeline_1_frozen_for_comparison": True,
        "selected_final_path": final_summary["selected_final_path"],
        "midi_range": [72, 76],
        "key_index_mapping": "key_index = midi_pitch - 21",
        "right_hand_only": True,
        "sustain": "none",
        "benchmark_sequences": [list(sequence) for sequence in BENCHMARK_SEQUENCE_PITCHES],
        "renderer": {
            "name": "FluidSynth",
            "version": "2.3.4",
            "soundfont": str(find_default_soundfont()),
            "sample_rate": 44100,
        },
        "transcription_model": _transcriber_dependency(args.transcriber),
        "confidence_threshold": args.confidence_threshold,
        "cleanup_enabled": final_summary["selected_final_path"] == "CLEANED",
        "cleanup_rules": {
            "invalid_notes": "discard zero/negative duration and optionally clip to audio duration",
            "same_pitch_duplicates": "suppress later predictions for a pitch already present",
            "monophonic_overlap": "truncate previous note offset to next predicted onset",
            "oracle_reference_used": False,
        },
        "canonical_representation": "ala_pianist.music.timed_notes.TimedNote",
        "controller_checkpoint_path": str(controller_checkpoint),
        "controller_checkpoint_sha256": _sha256(controller_checkpoint),
        "oracle_metrics": final_summary["controller_summary"]["oracle"],
        "raw_metrics": {
            "transcription": final_summary["transcription_summary"],
            "controller": final_summary["controller_summary"]["raw_basic_pitch"],
        },
        "cleaned_metrics": {
            "transcription": final_summary["transcription_summary"],
            "controller": final_summary["controller_summary"]["cleaned_basic_pitch"],
        },
        "selected_final_metrics": (
            final_summary["controller_summary"]["cleaned_basic_pitch"]
            if final_summary["selected_final_path"] == "CLEANED"
            else final_summary["controller_summary"]["raw_basic_pitch"]
        ),
        "known_limitations": [
            "Basic Pitch note-off estimates remain imperfect after bounded cleanup.",
            "Frozen symbolic controller is imperfect under oracle MIDI.",
            "Benchmark is generated five-note monophonic audio, not a real dataset.",
        ],
        "benchmark_command": "python scripts/run_indirect_cleanup_comparison.py --benchmark-dir experiments/indirect_pipeline/five_note_rendered_benchmark",
        "output_dir": str(output_dir),
        "source_git_commit": _git_commit(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _delta_optional(left, right):
    if left is None or right is None or left == "" or right == "":
        return None
    return float(left) - float(right)


def _note_payload(note: TimedNote) -> dict:
    return {
        "pitch": note.pitch,
        "onset": note.onset,
        "offset": note.offset,
        "duration": note.duration,
        "confidence": note.confidence,
        "source": note.source,
        "metadata": note.metadata,
    }


if __name__ == "__main__":
    main()
