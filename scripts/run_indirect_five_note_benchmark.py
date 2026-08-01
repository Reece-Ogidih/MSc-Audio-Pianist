"""Run the five-note rendered-audio Pipeline 1 benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from ala_pianist.audio import BasicPitchTranscriber, GeneratedWavPeakTranscriber
from ala_pianist.audio.transcriber import OracleMidiTranscriber, TranscriptionOutput
from ala_pianist.evaluation import binary_key_vector, pressed_key_metrics, timestep_key_metrics
from ala_pianist.evaluation.transcription_metrics import transcription_note_metrics
from ala_pianist.music import (
    timed_notes_to_controller_sequence,
    timing_quantization_report,
)
from ala_pianist.pipelines.indirect import (
    BENCHMARK_SEQUENCE_PITCHES,
    IndirectPipelineConfig,
    RenderedBenchmarkItem,
    benchmark_sequence_name,
    create_rendered_benchmark,
    find_default_soundfont,
    note_events_to_timed_notes,
    write_benchmark_manifest,
    write_benchmark_sequence_midi,
    write_controller_midi_from_result,
    write_csv_rows,
)
from ala_pianist.pipelines.indirect import IndirectPipelineSymbolicResult
from ala_pianist.rl import DroQPolicy, GeneralOneHandGoalEnv


ROOT = Path("/home/reece_dev/msc-audio-pianist")
DEFAULT_CONTROLLER = (
    ROOT
    / "artifacts"
    / "frozen_models"
    / "five_note_symbolic_controller_v1"
    / "checkpoint_800000_steps.pt"
)


def main() -> None:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcriber", default="basic_pitch", choices=["basic_pitch", "generated_wav_peak"])
    parser.add_argument("--midi-min", type=int, default=72)
    parser.add_argument("--midi-max", type=int, default=76)
    parser.add_argument("--output-dir", default=str(ROOT / "experiments" / "indirect_pipeline" / "five_note_rendered_benchmark"))
    parser.add_argument("--controller-checkpoint", default=str(DEFAULT_CONTROLLER))
    parser.add_argument("--render-audio", action="store_true")
    parser.add_argument("--evaluate-transcription", action="store_true")
    parser.add_argument("--evaluate-controller", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument("--offset-tolerance", type=float, default=0.10)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--fluidsynth-gain", type=float, default=0.5)
    parser.add_argument("--horizon-steps", type=int, default=96)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--checkpoint-device", default="cpu")
    parser.add_argument("--basic-pitch-onset-threshold", type=float, default=0.5)
    parser.add_argument("--basic-pitch-frame-threshold", type=float, default=0.3)
    parser.add_argument("--basic-pitch-min-note-ms", type=float, default=80.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    soundfont = find_default_soundfont()
    items = _render_or_load_benchmark(
        output_dir,
        render_audio=args.render_audio,
        soundfont=soundfont,
        sample_rate=args.sample_rate,
        gain=args.fluidsynth_gain,
    )
    config = IndirectPipelineConfig(
        midi_min=args.midi_min,
        midi_max=args.midi_max,
        confidence_threshold=args.confidence_threshold,
        range_policy="drop",
        duplicate_policy="merge",
    )
    transcriber = _build_transcriber(args)

    predictions_path = output_dir / "transcription_predictions.json"
    transcription_rows = []
    transcription_payload: dict[str, Any] = {
        "transcriber": args.transcriber,
        "confidence_threshold": args.confidence_threshold,
        "onset_tolerance_seconds": args.onset_tolerance,
        "offset_tolerance_seconds": args.offset_tolerance,
        "sequences": {},
    }
    oracle_controller_rows = []
    predicted_controller_rows = []
    comparison_rows = []
    failed_sequences = []

    policy = None
    if args.evaluate_controller:
        policy = DroQPolicy.load(args.controller_checkpoint, device=args.checkpoint_device)

    for item in items:
        print(f"benchmark_sequence={item.sequence_name} pitches={list(item.pitches)}")
        oracle_output = OracleMidiTranscriber(item.midi_path).transcribe(item.wav_path)
        predicted_output = _safe_transcribe(transcriber, item.wav_path)
        transcription_payload["sequences"][item.sequence_name] = {
            "reference_notes": [_note_payload(note) for note in item.notes],
            "raw_predicted_notes": [_note_payload(note) for note in predicted_output.notes],
            "predicted_metadata": predicted_output.metadata,
        }
        oracle_symbolic = _symbolic_result(oracle_output, item.notes, config, args)
        predicted_symbolic = _symbolic_result(predicted_output, item.notes, config, args)

        oracle_goal_midi = write_controller_midi_from_result(
            oracle_symbolic,
            output_dir / "controller_goals" / "oracle" / f"{item.sequence_name}.mid",
        )
        predicted_goal_midi = None
        if predicted_symbolic.controller_sequence.notes:
            predicted_goal_midi = write_controller_midi_from_result(
                predicted_symbolic,
                output_dir / "controller_goals" / "predicted" / f"{item.sequence_name}.mid",
            )

        if args.evaluate_transcription and predicted_symbolic.transcription_metrics is not None:
            with_offset_metrics = transcription_note_metrics(
                item.notes,
                predicted_symbolic.controller_sequence.notes,
                onset_tolerance_seconds=args.onset_tolerance,
                offset_tolerance_seconds=args.offset_tolerance,
            )
            loose_onset_metrics = transcription_note_metrics(
                item.notes,
                predicted_symbolic.controller_sequence.notes,
                onset_tolerance_seconds=max(args.onset_tolerance * 2.0, args.onset_tolerance),
                offset_tolerance_seconds=None,
            )
            row = {
                "sequence_name": item.sequence_name,
                "pitches": "-".join(map(str, item.pitches)),
                **predicted_symbolic.transcription_metrics.as_dict(),
                "with_offset_note_precision": with_offset_metrics.note_precision,
                "with_offset_note_recall": with_offset_metrics.note_recall,
                "with_offset_note_f1": with_offset_metrics.note_f1,
                "loose_onset_tolerance_seconds": max(args.onset_tolerance * 2.0, args.onset_tolerance),
                "loose_onset_note_precision": loose_onset_metrics.note_precision,
                "loose_onset_note_recall": loose_onset_metrics.note_recall,
                "loose_onset_note_f1": loose_onset_metrics.note_f1,
            }
            transcription_rows.append(_json_safe_row(row))

        if args.evaluate_controller and policy is not None:
            oracle_metrics = evaluate_controller_sequence(
                controller_midi_path=oracle_goal_midi,
                reference_notes=item.notes,
                policy=policy,
                seed=args.seed,
                horizon_steps=args.horizon_steps,
                midi_min=args.midi_min,
                midi_max=args.midi_max,
                condition="oracle",
                sequence_name=item.sequence_name,
            )
            oracle_controller_rows.append(oracle_metrics)
            if predicted_goal_midi is not None:
                predicted_metrics = evaluate_controller_sequence(
                    controller_midi_path=predicted_goal_midi,
                    reference_notes=item.notes,
                    policy=policy,
                    seed=args.seed,
                    horizon_steps=args.horizon_steps,
                    midi_min=args.midi_min,
                    midi_max=args.midi_max,
                    condition="predicted",
                    sequence_name=item.sequence_name,
                )
            else:
                predicted_metrics = _empty_controller_metrics(item, "predicted")
                failed_sequences.append(
                    {
                        "sequence_name": item.sequence_name,
                        "reason": "no predicted notes after range/confidence filtering",
                    }
                )
            predicted_controller_rows.append(predicted_metrics)
            comparison_rows.append(_compare_controller_rows(oracle_metrics, predicted_metrics))

    predictions_path.write_text(json.dumps(transcription_payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv_rows(transcription_rows, output_dir / "transcription_per_sequence.csv")
    write_csv_rows(oracle_controller_rows, output_dir / "controller_oracle_per_sequence.csv")
    write_csv_rows(predicted_controller_rows, output_dir / "controller_predicted_per_sequence.csv")
    write_csv_rows(comparison_rows, output_dir / "controller_comparison.csv")

    transcription_summary = _summary_rows(transcription_rows, "transcription")
    controller_oracle_summary = _summary_rows(oracle_controller_rows, "oracle_controller")
    controller_predicted_summary = _summary_rows(predicted_controller_rows, "predicted_controller")
    comparison_summary = _summary_rows(comparison_rows, "oracle_minus_predicted")
    runtime_seconds = time.perf_counter() - start_time
    summary = {
        "pipeline_1_architecturally_complete": _architectural_completion(
            args,
            items,
            predicted_controller_rows,
            failed_sequences,
        ),
        "performance_assessment": _performance_assessment(predicted_controller_rows),
        "runtime_seconds": runtime_seconds,
        "controller_checkpoint": str(args.controller_checkpoint),
        "transcriber": args.transcriber,
        "transcriber_dependency": _transcriber_dependency(args.transcriber),
        "soundfont": str(soundfont),
        "sample_rate": args.sample_rate,
        "confidence_threshold": args.confidence_threshold,
        "onset_tolerance_seconds": args.onset_tolerance,
        "offset_tolerance_seconds": args.offset_tolerance,
        "benchmark_sequences": [list(item.pitches) for item in items],
        "failed_sequences": failed_sequences,
        "transcription_summary": transcription_summary,
        "controller_oracle_summary": controller_oracle_summary,
        "controller_predicted_summary": controller_predicted_summary,
        "controller_degradation_summary": comparison_summary,
    }
    (output_dir / "transcription_summary.json").write_text(
        json.dumps(transcription_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "end_to_end_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir / "pipeline_1_v1_report.md", args, summary)
    print(f"output_dir={output_dir}")
    print(f"pipeline_1_architecturally_complete={summary['pipeline_1_architecturally_complete']}")
    print(f"performance_assessment={summary['performance_assessment']}")
    print(f"runtime_seconds={runtime_seconds:.2f}")


def _render_or_load_benchmark(
    output_dir: Path,
    *,
    render_audio: bool,
    soundfont: Path,
    sample_rate: int,
    gain: float,
) -> tuple[RenderedBenchmarkItem, ...]:
    if render_audio or not (output_dir / "benchmark_manifest.json").exists():
        return create_rendered_benchmark(
            output_dir,
            soundfont_path=soundfont,
            sample_rate=sample_rate,
            gain=gain,
        )
    manifest = json.loads((output_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    items = []
    for sequence in manifest["sequences"]:
        notes = tuple(
            note_events_to_timed_notes(
                [
                    type(
                        "Event",
                        (),
                        {
                            "pitch": pitch,
                            "start": onset,
                            "duration": duration,
                        },
                    )()
                    for pitch, onset, duration in zip(
                        sequence["midi_pitches"],
                        sequence["onset_times"],
                        sequence["durations"],
                    )
                ]
            )
        )
        items.append(
            RenderedBenchmarkItem(
                sequence["sequence_name"],
                tuple(sequence["midi_pitches"]),
                Path(sequence["midi_path"]),
                Path(sequence["wav_path"]),
                notes,
            )
        )
    return tuple(items)


def _build_transcriber(args) -> Any:
    if args.transcriber == "basic_pitch":
        return BasicPitchTranscriber(
            onset_threshold=args.basic_pitch_onset_threshold,
            frame_threshold=args.basic_pitch_frame_threshold,
            minimum_note_length_ms=args.basic_pitch_min_note_ms,
            minimum_frequency=500.0,
            maximum_frequency=1100.0,
        )
    return GeneratedWavPeakTranscriber()


def _safe_transcribe(transcriber, wav_path: Path) -> TranscriptionOutput:
    try:
        return transcriber.transcribe(wav_path)
    except Exception as exc:
        return TranscriptionOutput(
            notes=(),
            transcriber_name=getattr(transcriber, "name", type(transcriber).__name__),
            source_audio_path=wav_path,
            metadata={"error": f"{type(exc).__name__}: {exc}"},
        )


def _symbolic_result(
    output: TranscriptionOutput,
    expected_notes,
    config: IndirectPipelineConfig,
    args,
) -> IndirectPipelineSymbolicResult:
    sequence = timed_notes_to_controller_sequence(
        output.notes,
        midi_min=config.midi_min,
        midi_max=config.midi_max,
        confidence_threshold=config.confidence_threshold,
        range_policy="drop",
        duplicate_policy="merge",
        allow_polyphony=True,
    )
    metrics = transcription_note_metrics(
        expected_notes,
        sequence.notes,
        onset_tolerance_seconds=args.onset_tolerance,
        offset_tolerance_seconds=None,
    )
    return IndirectPipelineSymbolicResult(output, sequence, metrics)


def evaluate_controller_sequence(
    *,
    controller_midi_path: Path,
    reference_notes,
    policy,
    seed: int,
    horizon_steps: int,
    midi_min: int,
    midi_max: int,
    condition: str,
    sequence_name: str,
) -> dict:
    env = GeneralOneHandGoalEnv(
        midi_path=controller_midi_path,
        midi_min=midi_min,
        midi_max=midi_max,
        seed=seed,
        lookahead=1,
        horizon_steps=horizon_steps,
        action_mode="direct",
        action_repeat=1,
        reward_config=None,
    )
    obs, info = env.reset(seed=seed)
    del info
    control_timestep = _control_timestep(env)
    reference_keys = {note.key_index for note in reference_notes}
    pressed_keys = set()
    target_vectors = []
    pressed_vectors = []
    max_target = 0.0
    max_unintended = 0.0
    integrated_unintended = 0.0
    shaped_return = 0.0
    native_reward_sum = 0.0
    wrong_key_crossings = 0
    previous_pressed: set[int] = set()
    action_deltas = []
    action_saturation = []
    previous_action = np.zeros(22, dtype=np.float32)
    key_state_max = {key: 0.0 for key in range(88)}
    second_target_completed = 0.0
    previous_late_release_steps = 0
    events = tuple(reference_notes)
    for step in range(horizon_steps):
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        shaped_return += float(reward)
        native_reward_sum += float(info["native_reward"])
        states = env.piano_key_states()
        for key in range(states.shape[0]):
            key_state_max[key] = max(key_state_max[key], float(states[key]))
        current_pressed = set(info["pressed_keys"])
        pressed_keys.update(current_pressed)
        wrong_key_crossings += len([key for key in current_pressed - previous_pressed if key not in reference_keys])
        previous_pressed = current_pressed
        active_ref = _reference_active_keys(events, step * control_timestep)
        target_vectors.append(binary_key_vector(active_ref))
        pressed_vectors.append(binary_key_vector(current_pressed))
        target_states = [float(states[key]) for key in reference_keys if 0 <= key < states.shape[0]]
        max_target = max(max_target, max(target_states, default=0.0))
        unintended_states = [
            float(states[key]) for key in range(states.shape[0]) if key not in reference_keys
        ]
        step_unintended = max(unintended_states, default=0.0)
        max_unintended = max(max_unintended, step_unintended)
        integrated_unintended += step_unintended * control_timestep
        action_deltas.append(float(np.mean(np.abs(action - previous_action))))
        action_saturation.append(float(np.mean(np.abs(action) >= 0.95)))
        previous_action = action.copy()
        if len(events) >= 2:
            second = events[1].key_index
            if current_pressed and second in current_pressed:
                second_target_completed = 1.0
            first = events[0].key_index
            if step * control_timestep >= events[0].offset and first in current_pressed:
                previous_late_release_steps += 1
        if terminated or truncated:
            break
    pressed_metrics = pressed_key_metrics(reference_keys, pressed_keys)
    timestep_metrics = timestep_key_metrics(target_vectors, pressed_vectors)
    active_goal_sequence = timed_notes_to_controller_sequence(
        reference_notes,
        midi_min=midi_min,
        midi_max=midi_max,
        range_policy="drop",
    )
    quantization = timing_quantization_report(
        active_goal_sequence,
        control_timestep_seconds=control_timestep,
    )
    max_key_states_for_targets = {
        int(key): float(key_state_max.get(int(key), 0.0)) for key in sorted(reference_keys)
    }
    return _json_safe_row(
        {
            "condition": condition,
            "sequence_name": sequence_name,
            "controller_midi_path": str(controller_midi_path),
            "reference_pitches": "-".join(str(note.pitch) for note in events),
            "reference_keys": "-".join(str(key) for key in sorted(reference_keys)),
            "pressed_keys": "-".join(str(key) for key in sorted(pressed_keys)),
            "pressed_key_precision": pressed_metrics.precision,
            "pressed_key_recall": pressed_metrics.recall,
            "pressed_key_f1": pressed_metrics.f1,
            "timestep_precision": timestep_metrics.precision,
            "timestep_recall": timestep_metrics.recall,
            "timestep_f1": timestep_metrics.f1,
            "max_target_key_state": max_target,
            "max_unintended_key_state": max_unintended,
            "integrated_unintended_key_state": integrated_unintended,
            "wrong_key_crossings": wrong_key_crossings,
            "transition_completion": 1.0
            if len(events) < 2
            else float(all(key in pressed_keys for key in reference_keys)),
            "second_target_completion": second_target_completed if len(events) >= 2 else None,
            "previous_target_late_release_duration": previous_late_release_steps * control_timestep,
            "shaped_return": shaped_return,
            "native_reward_sum": native_reward_sum,
            "mean_abs_action_delta": float(np.mean(action_deltas)) if action_deltas else 0.0,
            "action_saturation_fraction": float(np.mean(action_saturation)) if action_saturation else 0.0,
            "control_timestep_seconds": control_timestep,
            "timing_quantization_max_error_seconds": quantization["max_abs_error_seconds"],
            "target_key_state_by_key": json.dumps(max_key_states_for_targets, sort_keys=True),
            "strict_outcome": _strict_outcome(reference_keys, pressed_keys, max_target, max_unintended),
        }
    )


def _empty_controller_metrics(item: RenderedBenchmarkItem, condition: str) -> dict:
    keys = {pitch - 21 for pitch in item.pitches}
    pressed_metrics = pressed_key_metrics(keys, set())
    return _json_safe_row(
        {
            "condition": condition,
            "sequence_name": item.sequence_name,
            "controller_midi_path": None,
            "reference_pitches": "-".join(str(pitch) for pitch in item.pitches),
            "reference_keys": "-".join(str(key) for key in sorted(keys)),
            "pressed_keys": "",
            "pressed_key_precision": pressed_metrics.precision,
            "pressed_key_recall": pressed_metrics.recall,
            "pressed_key_f1": pressed_metrics.f1,
            "timestep_precision": 0.0,
            "timestep_recall": 0.0,
            "timestep_f1": 0.0,
            "max_target_key_state": 0.0,
            "max_unintended_key_state": 0.0,
            "integrated_unintended_key_state": 0.0,
            "wrong_key_crossings": 0,
            "transition_completion": 0.0,
            "second_target_completion": 0.0 if len(item.pitches) >= 2 else None,
            "previous_target_late_release_duration": 0.0,
            "shaped_return": 0.0,
            "native_reward_sum": 0.0,
            "mean_abs_action_delta": 0.0,
            "action_saturation_fraction": 0.0,
            "control_timestep_seconds": None,
            "timing_quantization_max_error_seconds": None,
            "target_key_state_by_key": "{}",
            "strict_outcome": "no_predicted_goal",
        }
    )


def _compare_controller_rows(oracle: dict, predicted: dict) -> dict:
    keys = [
        "pressed_key_precision",
        "pressed_key_recall",
        "pressed_key_f1",
        "timestep_precision",
        "timestep_recall",
        "timestep_f1",
        "max_target_key_state",
        "max_unintended_key_state",
        "integrated_unintended_key_state",
        "wrong_key_crossings",
        "transition_completion",
        "second_target_completion",
        "previous_target_late_release_duration",
        "shaped_return",
        "native_reward_sum",
    ]
    row = {"sequence_name": oracle["sequence_name"]}
    for key in keys:
        left = oracle.get(key)
        right = predicted.get(key)
        row[f"oracle_{key}"] = left
        row[f"predicted_{key}"] = right
        row[f"delta_predicted_minus_oracle_{key}"] = (
            None if left is None or right is None else float(right) - float(left)
        )
    row["oracle_strict_outcome"] = oracle.get("strict_outcome")
    row["predicted_strict_outcome"] = predicted.get("strict_outcome")
    row["oracle_pressed_keys"] = oracle.get("pressed_keys")
    row["predicted_pressed_keys"] = predicted.get("pressed_keys")
    return _json_safe_row(row)


def _control_timestep(env: GeneralOneHandGoalEnv) -> float:
    try:
        return float(env.env.control_timestep())
    except Exception:
        return 0.05


def _reference_active_keys(notes, time_seconds: float) -> set[int]:
    return {note.key_index for note in notes if note.onset <= time_seconds < note.offset}


def _strict_outcome(target_keys, pressed_keys, max_target, max_unintended) -> str:
    target_keys = set(target_keys)
    pressed_keys = set(pressed_keys)
    if target_keys and target_keys.issubset(pressed_keys) and not (pressed_keys - target_keys):
        return "clean_low_unintended" if max_unintended < 0.25 else "clean_high_unintended"
    if target_keys and target_keys.intersection(pressed_keys):
        return "dirty_pressed_wrong_key"
    if max_target >= 0.25 and max_unintended < 0.25:
        return "near_clean_partial"
    return "missed"


def _note_payload(note) -> dict:
    return {
        "pitch": int(note.pitch),
        "key_index": int(note.key_index),
        "onset": float(note.onset),
        "offset": float(note.offset),
        "duration": float(note.duration),
        "confidence": float(note.confidence),
        "source": str(note.source),
        "metadata": note.metadata,
    }


def _json_safe_row(row: dict) -> dict:
    safe = {}
    for key, value in row.items():
        if isinstance(value, (list, tuple, dict)):
            safe[key] = json.dumps(value, sort_keys=True)
        elif isinstance(value, np.generic):
            safe[key] = value.item()
        else:
            safe[key] = value
    return safe


def _summary_rows(rows: list[dict], prefix: str) -> dict:
    if not rows:
        return {"prefix": prefix, "row_count": 0}
    numeric_keys = []
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if all(value is None or isinstance(value, (int, float)) for value in values):
            numeric_keys.append(key)
    summary = {"prefix": prefix, "row_count": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_min"] = float(np.min(values))
            summary[f"{key}_max"] = float(np.max(values))
    return summary


def _architectural_completion(args, items, predicted_controller_rows, failed_sequences) -> bool:
    all_pitches = {pitch for item in items for pitch in item.pitches}
    has_predicted_controller_rows = bool(predicted_controller_rows)
    saw_robot_execution = any(row.get("strict_outcome") != "no_predicted_goal" for row in predicted_controller_rows)
    return bool(
        args.transcriber != "generated_wav_peak"
        and all_pitches == set(range(args.midi_min, args.midi_max + 1))
        and len(items) == len(BENCHMARK_SEQUENCE_PITCHES)
        and args.evaluate_controller
        and has_predicted_controller_rows
        and saw_robot_execution
        and len(failed_sequences) < len(items)
    )


def _performance_assessment(predicted_controller_rows) -> str:
    if not predicted_controller_rows:
        return "non-functional"
    f1_values = [float(row.get("pressed_key_f1", 0.0)) for row in predicted_controller_rows]
    mean_f1 = float(np.mean(f1_values))
    if mean_f1 >= 0.85:
        return "strong"
    if mean_f1 >= 0.55:
        return "usable baseline"
    if mean_f1 > 0.0:
        return "weak but functional"
    return "non-functional"


def _transcriber_dependency(name: str) -> dict:
    if name != "basic_pitch":
        return {"name": "generated_wav_peak", "version": "project-local"}
    try:
        from importlib.metadata import version
        from basic_pitch import inference as basic_pitch_inference

        return {
            "name": "basic-pitch",
            "version": version("basic-pitch"),
            "runtime": "onnx",
            "onnxruntime_version": version("onnxruntime"),
            "model_path": str(Path(basic_pitch_inference.ICASSP_2022_MODEL_PATH).with_suffix(".onnx")),
        }
    except Exception as exc:
        return {"name": "basic-pitch", "error": str(exc)}


def _write_report(path: Path, args, summary: dict) -> None:
    lines = [
        "# Pipeline 1 Five-Note Audio-to-Action Benchmark",
        "",
        "## Setup",
        "",
        f"- transcriber: `{args.transcriber}`",
        f"- controller checkpoint: `{args.controller_checkpoint}`",
        f"- MIDI range: `{args.midi_min}-{args.midi_max}`",
        f"- confidence threshold: `{args.confidence_threshold}`",
        f"- onset tolerance: `{args.onset_tolerance}` seconds",
        f"- offset tolerance: `{args.offset_tolerance}` seconds",
        f"- soundfont: `{summary['soundfont']}`",
        "",
        "## Completion Gate",
        "",
        f"`pipeline_1_architecturally_complete`: `{str(summary['pipeline_1_architecturally_complete']).lower()}`",
        "",
        f"Performance assessment: `{summary['performance_assessment']}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "Detailed CSV files are written beside this report.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
