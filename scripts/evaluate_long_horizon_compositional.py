#!/usr/bin/env python3
"""Evaluate zero-shot long-horizon Audio-to-Action composition."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from ala_pianist.audio import BasicPitchTranscriber, GeneratedWavPeakTranscriber
from ala_pianist.audio.transcriber import OracleMidiTranscriber, TranscriptionOutput
from ala_pianist.evaluation import (
    binary_key_vector,
    horizon_steps_for_notes,
    load_long_horizon_benchmark,
    long_horizon_metrics_from_trace,
    sequence_counts_by_length_and_archetype,
    sequence_notes,
)
from ala_pianist.evaluation.direct_audio import sha256_file, strict_outcome
from ala_pianist.evaluation.final_experiments import write_csv
from ala_pianist.evaluation.transcription_metrics import transcription_note_metrics
from ala_pianist.music import timed_notes_to_controller_sequence, write_sequence_midi
from ala_pianist.music.timed_notes import TimedNote
from ala_pianist.pipelines.indirect import (
    IndirectPipelineConfig,
    RenderedBenchmarkItem,
    find_default_soundfont,
    render_midi_with_fluidsynth,
    write_controller_midi_from_result,
)
from ala_pianist.pipelines.indirect import IndirectPipelineSymbolicResult
from ala_pianist.rl import DirectAudioGoalEnv, DirectDroQAgent, DroQPolicy, GeneralOneHandGoalEnv
from ala_pianist.rl.direct_audio_env import build_direct_audio_reference_bank


ROOT = Path("/home/reece_dev/msc-audio-pianist")
DEFAULT_MANIFEST = ROOT / "configs" / "long_horizon_compositional_v1.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "long_horizon_compositional_v1"
DEFAULT_SYMBOLIC_CONTROLLER = (
    ROOT / "artifacts" / "frozen_models" / "five_note_symbolic_controller_v1" / "checkpoint_800000_steps.pt"
)
PIPELINE2_CHECKPOINTS = {
    "pipeline2_seed13_1m": ROOT
    / "artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/"
    / "lightweight_checkpoints/checkpoint_1000000_steps.pt",
    "pipeline2_seed61_1m": ROOT
    / "artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/"
    / "lightweight_checkpoints/checkpoint_1000000_steps.pt",
    "pipeline2_seed13_1p25m": ROOT
    / "artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed13_resume_1m_to_1p5m_retry1/"
    / "lightweight_checkpoints/checkpoint_1250000_steps.pt",
}


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-short-primitives", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--pipeline1-transcriber", choices=["basic_pitch", "generated_wav_peak"], default="basic_pitch")
    parser.add_argument(
        "--pipeline1-condition",
        choices=["both", "oracle", "transcribed"],
        default="both",
        help="Select Pipeline 1 conditions without recomputing an already valid condition.",
    )
    parser.add_argument("--pipeline1-controller", type=Path, default=DEFAULT_SYMBOLIC_CONTROLLER)
    parser.add_argument("--skip-pipeline1", action="store_true")
    parser.add_argument("--skip-pipeline2", action="store_true")
    parser.add_argument("--pipeline2-model", action="append", default=None)
    parser.add_argument(
        "--pipeline2-checkpoint-path",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional Pipeline 2 checkpoint path, usable with --pipeline2-model LABEL.",
    )
    parser.add_argument("--include-audio-interventions", action="store_true")
    parser.add_argument(
        "--audio-intervention-sequence",
        action="append",
        default=[],
        metavar="MIDI[,MIDI...]",
        help="Run zero/mismatched rollouts for this exact sequence; repeat as needed.",
    )
    parser.add_argument(
        "--sequence-name",
        action="append",
        default=[],
        help="Evaluate only this exact manifest sequence name; repeat as needed.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--fluidsynth-gain", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--onset-tolerance", type=float, default=0.05)
    parser.add_argument("--offset-tolerance", type=float, default=0.10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = load_long_horizon_benchmark(
        args.manifest,
        allow_trained_short=args.allow_short_primitives,
    )
    sequences = list(benchmark.sequences)
    if args.sequence_name:
        requested_names = set(args.sequence_name)
        available_names = {sequence.name for sequence in sequences}
        unknown_names = requested_names - available_names
        if unknown_names:
            raise ValueError(f"Unknown benchmark sequence names: {sorted(unknown_names)}")
        sequences = [sequence for sequence in sequences if sequence.name in requested_names]
    if args.smoke:
        sequences = [sequence for sequence in sequences if sequence.length in {3, 5}][:2]
        args.pipeline2_model = args.pipeline2_model or ["pipeline2_seed13_1m"]
        args.include_audio_interventions = False
    if args.max_sequences is not None:
        sequences = sequences[: int(args.max_sequences)]
    if not sequences:
        raise ValueError("No long-horizon sequences selected.")

    pipeline2_checkpoints = _pipeline2_checkpoint_specs(args.pipeline2_checkpoint_path)
    checkpoint_labels = [] if args.skip_pipeline2 else (args.pipeline2_model or list(pipeline2_checkpoints))
    checkpoint_audit = _audit_checkpoints(
        checkpoint_labels,
        pipeline1_controller=args.pipeline1_controller,
        pipeline2_checkpoints=pipeline2_checkpoints,
        include_pipeline1=not args.skip_pipeline1,
    )
    soundfont = find_default_soundfont()
    items = _render_benchmark_items(
        sequences,
        benchmark_timing=benchmark.timing,
        output_dir=output_dir / "rendered_benchmark",
        soundfont=soundfont,
        sample_rate=args.sample_rate,
        gain=args.fluidsynth_gain,
    )
    selected_sequences = tuple(tuple(item.pitches) for item in items)
    max_horizon = max(horizon_steps_for_notes(item.notes) for item in items)

    sequence_rows: list[dict[str, Any]] = []
    transcription_rows: list[dict[str, Any]] = []
    audio_dependence_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    if not args.skip_pipeline1:
        p1_sequence_rows, p1_transcription_rows, p1_event_rows = _evaluate_pipeline1(
            items=items,
            output_dir=output_dir / "pipeline1",
            controller_checkpoint=args.pipeline1_controller,
            transcriber_name=args.pipeline1_transcriber,
            condition=args.pipeline1_condition,
            config=IndirectPipelineConfig(
                midi_min=min(benchmark.midi_pitches),
                midi_max=max(benchmark.midi_pitches),
                confidence_threshold=args.confidence_threshold,
                range_policy="drop",
                duplicate_policy="merge",
            ),
            horizon_steps=max_horizon,
            seed=args.seed,
            device=args.device,
            onset_tolerance=args.onset_tolerance,
            offset_tolerance=args.offset_tolerance,
        )
        sequence_rows.extend(p1_sequence_rows)
        transcription_rows.extend(p1_transcription_rows)
        event_rows.extend(p1_event_rows)

    if not args.skip_pipeline2:
        intervention_sequences = _parse_sequence_specs(args.audio_intervention_sequence)
        p2_sequence_rows, p2_audio_rows, p2_event_rows = _evaluate_pipeline2(
            sequences=selected_sequences,
            items=items,
            output_dir=output_dir / "pipeline2",
            checkpoint_labels=checkpoint_labels,
            pipeline2_checkpoints=pipeline2_checkpoints,
            horizon_steps=max_horizon,
            seed=args.seed,
            device=args.device,
            include_audio_interventions=args.include_audio_interventions,
            intervention_sequences=intervention_sequences,
        )
        sequence_rows.extend(p2_sequence_rows)
        audio_dependence_rows.extend(p2_audio_rows)
        event_rows.extend(p2_event_rows)

    length_rows = _aggregate_rows(sequence_rows, ["model_label", "audio_mode", "sequence_length"])
    model_rows = _aggregate_rows(sequence_rows, ["model_label", "audio_mode"])
    archetype_rows = _aggregate_rows(sequence_rows, ["model_label", "audio_mode", "sequence_length", "archetype"])
    manifest_payload = {
        "benchmark_name": benchmark.benchmark_name,
        "manifest_path": str(args.manifest),
        "sequence_count": len(sequences),
        "sequence_counts_by_length_and_archetype": sequence_counts_by_length_and_archetype(sequences),
        "sequences": [_sequence_payload(sequence, benchmark.timing) for sequence in sequences],
        "timing": asdict(benchmark.timing),
        "fairness_audit": _fairness_audit(max_horizon=max_horizon, benchmark=benchmark),
        "checkpoint_audit": checkpoint_audit,
        "soundfont": str(soundfont),
    }
    summary = {
        "benchmark_name": benchmark.benchmark_name,
        "runtime_seconds": time.perf_counter() - started,
        "sequence_count": len(sequences),
        "models": sorted({str(row["model_label"]) for row in sequence_rows}),
        "outputs": {
            "long_horizon_sequence_metrics": str(output_dir / "long_horizon_sequence_metrics.csv"),
            "long_horizon_length_summary": str(output_dir / "long_horizon_length_summary.csv"),
            "long_horizon_model_summary": str(output_dir / "long_horizon_model_summary.csv"),
            "pipeline1_transcription_long_horizon": str(output_dir / "pipeline1_transcription_long_horizon.csv"),
            "audio_dependence_long_horizon": str(output_dir / "audio_dependence_long_horizon.csv"),
        },
    }
    write_csv(output_dir / "long_horizon_sequence_metrics.csv", _json_safe_rows(sequence_rows))
    write_csv(output_dir / "long_horizon_event_metrics.csv", _json_safe_rows(event_rows))
    write_csv(output_dir / "long_horizon_length_summary.csv", _json_safe_rows(length_rows))
    write_csv(output_dir / "long_horizon_model_summary.csv", _json_safe_rows(model_rows))
    write_csv(output_dir / "long_horizon_archetype_summary.csv", _json_safe_rows(archetype_rows))
    write_csv(output_dir / "pipeline1_transcription_long_horizon.csv", _json_safe_rows(transcription_rows))
    write_csv(output_dir / "audio_dependence_long_horizon.csv", _json_safe_rows(audio_dependence_rows))
    (output_dir / "long_horizon_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "long_horizon_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"output_dir={output_dir}")
    print(f"sequence_count={len(sequences)}")
    print(f"max_horizon_steps={max_horizon}")
    print(f"runtime_seconds={summary['runtime_seconds']:.2f}")
    print("LONG_HORIZON_EVALUATION_COMPLETE=true")


def _pipeline2_checkpoint_specs(extra_specs: list[str]) -> dict[str, Path]:
    specs = dict(PIPELINE2_CHECKPOINTS)
    for spec in extra_specs:
        if "=" not in spec:
            raise ValueError(f"--pipeline2-checkpoint-path must be LABEL=PATH, got {spec!r}.")
        label, path = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("--pipeline2-checkpoint-path label must not be empty.")
        specs[label] = Path(path)
    return specs


def _audit_checkpoints(
    selected_labels: list[str],
    *,
    pipeline1_controller: Path,
    pipeline2_checkpoints: dict[str, Path],
    include_pipeline1: bool = True,
) -> dict[str, Any]:
    labels = selected_labels
    audit = {
        "pipeline1_symbolic_controller": {
            "path": str(pipeline1_controller),
            "exists": pipeline1_controller.is_file(),
            "sha256": sha256_file(pipeline1_controller) if pipeline1_controller.is_file() else None,
        },
        "pipeline2": {},
    }
    if include_pipeline1 and not pipeline1_controller.is_file():
        raise FileNotFoundError(f"Pipeline 1 controller not found: {pipeline1_controller}")
    for label in labels:
        path = pipeline2_checkpoints[label]
        if not path.is_file():
            raise FileNotFoundError(f"Pipeline 2 checkpoint {label} is missing: {path}")
        agent = DirectDroQAgent.load(path, device="cpu")
        audit["pipeline2"][label] = {
            "path": str(path),
            "exists": True,
            "sha256": sha256_file(path),
            "action_dim": agent.config.action_dim,
            "audio_window_size": agent.config.audio_window_size,
            "physical_dim": agent.config.physical_dim,
        }
    return audit


def _render_benchmark_items(
    sequences,
    *,
    benchmark_timing,
    output_dir: Path,
    soundfont: Path,
    sample_rate: int,
    gain: float,
) -> tuple[RenderedBenchmarkItem, ...]:
    items = []
    midi_dir = output_dir / "midi"
    wav_dir = output_dir / "wav"
    for sequence in sequences:
        notes = sequence_notes(sequence, benchmark_timing)
        midi_path = midi_dir / f"{sequence.name}.mid"
        wav_path = wav_dir / f"{sequence.name}.wav"
        if not midi_path.exists():
            write_sequence_midi(
                sequence.pitches,
                midi_path,
                midi_min=min(sequence.pitches),
                midi_max=max(sequence.pitches),
                timing=benchmark_timing,
                title=f"ALA Pianist {sequence.name}",
            )
        if not wav_path.exists():
            render_midi_with_fluidsynth(
                midi_path,
                wav_path,
                soundfont_path=soundfont,
                sample_rate=sample_rate,
                gain=gain,
            )
        items.append(RenderedBenchmarkItem(sequence.name, sequence.pitches, midi_path, wav_path, notes))
    return tuple(items)


def _evaluate_pipeline1(
    *,
    items: tuple[RenderedBenchmarkItem, ...],
    output_dir: Path,
    controller_checkpoint: Path,
    transcriber_name: str,
    condition: str,
    config: IndirectPipelineConfig,
    horizon_steps: int,
    seed: int,
    device: str,
    onset_tolerance: float,
    offset_tolerance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    policy = DroQPolicy.load(controller_checkpoint, device=device)
    if condition not in {"both", "oracle", "transcribed"}:
        raise ValueError(f"Unsupported Pipeline 1 condition: {condition!r}")
    transcriber = None if condition == "oracle" else _build_transcriber(transcriber_name)
    sequence_rows = []
    transcription_rows = []
    event_rows = []
    for item_index, item in enumerate(items):
        oracle_output = OracleMidiTranscriber(item.midi_path).transcribe(item.wav_path)
        outputs = []
        if condition in {"both", "oracle"}:
            outputs.append(("pipeline1_oracle", oracle_output))
        if condition in {"both", "transcribed"}:
            assert transcriber is not None
            outputs.append(("pipeline1_basic_pitch", _safe_transcribe(transcriber, item.wav_path)))
        for pipeline_condition, output in outputs:
            symbolic = _symbolic_result(
                output,
                item.notes,
                config,
                onset_tolerance=onset_tolerance,
                offset_tolerance=None if pipeline_condition == "pipeline1_oracle" else offset_tolerance,
            )
            if pipeline_condition == "pipeline1_basic_pitch":
                metric = symbolic.transcription_metrics
                transcription_rows.append(
                    {
                        "model_label": "Pipeline1 Basic Pitch",
                        "sequence_name": item.sequence_name,
                        "sequence": "-".join(str(pitch) for pitch in item.pitches),
                        "sequence_length": len(item.pitches),
                        "transcriber_name": predicted_output.transcriber_name,
                        "transcriber_error": predicted_output.metadata.get("error", ""),
                        **({} if metric is None else metric.as_dict()),
                    }
                )
            if not symbolic.controller_sequence.notes:
                sequence_rows.append(_empty_sequence_row(item, "Pipeline1 Basic Pitch", "no_predicted_goal"))
                continue
            goal_midi = write_controller_midi_from_result(
                symbolic,
                output_dir / "controller_goals" / pipeline_condition / f"{item.sequence_name}.mid",
            )
            row, events = _rollout_symbolic_policy(
                controller_midi_path=goal_midi,
                reference_notes=item.notes,
                policy=policy,
                seed=seed + item_index,
                horizon_steps=horizon_steps,
                midi_min=config.midi_min,
                midi_max=config.midi_max,
                model_label="Pipeline1 Oracle" if pipeline_condition == "pipeline1_oracle" else "Pipeline1 Basic Pitch",
                sequence_name=item.sequence_name,
                sequence=item.pitches,
                archetype=_archetype_from_name(item.sequence_name),
            )
            sequence_rows.append(row)
            event_rows.extend(events)
    return sequence_rows, transcription_rows, event_rows


def _evaluate_pipeline2(
    *,
    sequences: tuple[tuple[int, ...], ...],
    items: tuple[RenderedBenchmarkItem, ...],
    output_dir: Path,
    checkpoint_labels: list[str],
    pipeline2_checkpoints: dict[str, Path],
    horizon_steps: int,
    seed: int,
    device: str,
    include_audio_interventions: bool,
    intervention_sequences: tuple[tuple[int, ...], ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _bank, clips = build_direct_audio_reference_bank(
        generated_root=output_dir / "canonical_audio",
        sequences=sequences,
        variants_per_sequence=1,
        split="long_horizon_eval",
    )
    env = DirectAudioGoalEnv(
        audio_bank=_bank,
        clips=tuple(clips),
        sequences=sequences,
        sequence_sampling_weights=tuple(1.0 / len(sequences) for _ in sequences),
        horizon_steps=horizon_steps,
        lookahead=1,
        seed=seed,
        sampling_split="long_horizon_eval",
    )
    sequence_to_clip = {tuple(clip.sequence): index for index, clip in enumerate(env.clips)}
    notes_by_sequence = {tuple(item.pitches): item.notes for item in items}
    name_by_sequence = {tuple(item.pitches): item.sequence_name for item in items}
    modes_by_sequence = _audio_modes_by_sequence(
        sequences,
        include_audio_interventions=include_audio_interventions,
        intervention_sequences=intervention_sequences,
    )
    sequence_rows = []
    audio_rows = []
    event_rows = []
    for label in checkpoint_labels:
        path = pipeline2_checkpoints[label]
        agent = DirectDroQAgent.load(path, device=device)
        checkpoint_hash = sha256_file(path)
        for sequence_index, sequence in enumerate(sequences):
            clip_index = sequence_to_clip[tuple(sequence)]
            mismatch_sequence = _mismatch_sequence(sequence, sequences)
            mismatch_clip_id = env.clips[sequence_to_clip[mismatch_sequence]].clip_id
            audio_rows.append(_action_dependence_row(agent, env, sequence, clip_index, mismatch_sequence, mismatch_clip_id, label))
            for mode in modes_by_sequence[tuple(sequence)]:
                row, events = _rollout_direct_audio(
                    agent=agent,
                    env=env,
                    sequence=tuple(sequence),
                    notes=notes_by_sequence[tuple(sequence)],
                    sequence_name=name_by_sequence[tuple(sequence)],
                    clip_index=clip_index,
                    audio_mode=mode,
                    mismatch_sequence=mismatch_sequence,
                    mismatch_clip_id=mismatch_clip_id,
                    model_label=_pipeline2_label(label),
                    checkpoint_path=path,
                    checkpoint_hash=checkpoint_hash,
                    seed=seed + sequence_index,
                )
                sequence_rows.append(row)
                event_rows.extend(events)
    return sequence_rows, audio_rows, event_rows


def _rollout_symbolic_policy(
    *,
    controller_midi_path: Path,
    reference_notes: tuple[TimedNote, ...],
    policy: DroQPolicy,
    seed: int,
    horizon_steps: int,
    midi_min: int,
    midi_max: int,
    model_label: str,
    sequence_name: str,
    sequence: tuple[int, ...],
    archetype: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = GeneralOneHandGoalEnv(
        midi_path=controller_midi_path,
        midi_min=midi_min,
        midi_max=midi_max,
        seed=seed,
        lookahead=1,
        horizon_steps=horizon_steps,
        action_mode="direct",
        action_repeat=1,
    )
    obs, _info = env.reset(seed=seed)
    trace = _new_trace()
    shaped_return = 0.0
    native_reward_sum = 0.0
    saturation = []
    control_timestep = _control_timestep(env)
    for step in range(horizon_steps):
        action, _ = policy.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        shaped_return += float(reward)
        native_reward_sum += float(info.get("native_reward", 0.0))
        saturation.append(float(np.mean(np.abs(action) >= 0.95)))
        _append_trace(trace, reference_notes, step * control_timestep, env.piano_key_states(), info.get("pressed_keys", ()))
        if terminated or truncated:
            break
    return _trace_to_row(
        trace=trace,
        reference_notes=reference_notes,
        model_label=model_label,
        pipeline="pipeline1",
        sequence_name=sequence_name,
        sequence=sequence,
        archetype=archetype,
        audio_mode="oracle" if model_label == "Pipeline1 Oracle" else "basic_pitch",
        shaped_return=shaped_return,
        native_reward_sum=native_reward_sum,
        action_saturation=float(np.mean(saturation)) if saturation else 0.0,
        extra={"controller_midi_path": str(controller_midi_path)},
    )


def _rollout_direct_audio(
    *,
    agent: DirectDroQAgent,
    env: DirectAudioGoalEnv,
    sequence: tuple[int, ...],
    notes: tuple[TimedNote, ...],
    sequence_name: str,
    clip_index: int,
    audio_mode: str,
    mismatch_sequence: tuple[int, ...],
    mismatch_clip_id: int,
    model_label: str,
    checkpoint_path: Path,
    checkpoint_hash: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs, _info = env.reset_to_clip_index(clip_index, seed=seed)
    trace = _new_trace()
    shaped_return = 0.0
    native_reward_sum = 0.0
    saturation = []
    for step in range(env.horizon_steps):
        policy_obs = env.observation_for_audio_mode(obs, mode=audio_mode, mismatched_clip_id=mismatch_clip_id)
        action = agent.act(policy_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        base_env = env._base_env_for_clip_index(env._active_clip_index)
        shaped_return += float(reward)
        native_reward_sum += float(info.get("native_reward", 0.0))
        saturation.append(float(np.mean(np.abs(action) >= 0.95)))
        _append_trace(
            trace,
            notes,
            step * env.control_timestep_seconds,
            base_env.piano_key_states(),
            info.get("pressed_keys", ()),
        )
        if terminated or truncated:
            break
    return _trace_to_row(
        trace=trace,
        reference_notes=notes,
        model_label=model_label,
        pipeline="pipeline2",
        sequence_name=sequence_name,
        sequence=sequence,
        archetype=_archetype_from_name(sequence_name),
        audio_mode=audio_mode,
        shaped_return=shaped_return,
        native_reward_sum=native_reward_sum,
        action_saturation=float(np.mean(saturation)) if saturation else 0.0,
        extra={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "mismatched_sequence": "-".join(str(p) for p in mismatch_sequence) if audio_mode == "mismatched" else "",
        },
    )


def _trace_to_row(
    *,
    trace: dict[str, list],
    reference_notes: tuple[TimedNote, ...],
    model_label: str,
    pipeline: str,
    sequence_name: str,
    sequence: tuple[int, ...],
    archetype: str,
    audio_mode: str,
    shaped_return: float,
    native_reward_sum: float,
    action_saturation: float,
    extra: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = long_horizon_metrics_from_trace(
        notes=reference_notes,
        step_times=trace["times"],
        target_vectors=trace["target_vectors"],
        pressed_vectors=trace["pressed_vectors"],
        key_states=trace["key_states"],
    )
    target_keys = {note.key_index for note in reference_notes}
    pressed_keys = {
        int(key)
        for vector in trace["pressed_vectors"]
        for key in np.flatnonzero(np.asarray(vector, dtype=bool))
    }
    key_states = np.asarray(trace["key_states"], dtype=np.float32)
    row = {
        "model_label": model_label,
        "pipeline": pipeline,
        "sequence_name": sequence_name,
        "sequence": "-".join(str(pitch) for pitch in sequence),
        "sequence_length": len(sequence),
        "archetype": archetype,
        "audio_mode": audio_mode,
        "episode_duration_seconds": float(len(trace["times"]) * 0.05),
        "target_keys": "-".join(str(key) for key in sorted(target_keys)),
        "pressed_keys": "-".join(str(key) for key in sorted(pressed_keys)),
        "wrong_press_count": len(pressed_keys - target_keys),
        "max_target_key_state": _max_target_state(reference_notes, key_states),
        "max_unintended_key_state": _max_unintended_state(reference_notes, trace["times"], key_states),
        "integrated_unintended_key_state": _integrated_unintended_state(reference_notes, trace["times"], key_states),
        "shaped_return": float(shaped_return),
        "native_reward_sum": float(native_reward_sum),
        "mean_action_saturation": float(action_saturation),
        "strict_outcome": strict_outcome(target_keys, pressed_keys, _max_target_state(reference_notes, key_states), _max_unintended_state(reference_notes, trace["times"], key_states)),
        **{key: value for key, value in metrics.items() if key != "event_rows"},
        **extra,
    }
    event_rows = [
        {
            "model_label": model_label,
            "pipeline": pipeline,
            "sequence_name": sequence_name,
            "sequence_length": len(sequence),
            "archetype": archetype,
            "audio_mode": audio_mode,
            **event,
        }
        for event in metrics["event_rows"]
    ]
    return row, event_rows


def _new_trace() -> dict[str, list]:
    return {"times": [], "target_vectors": [], "pressed_vectors": [], "key_states": []}


def _append_trace(trace: dict[str, list], notes: tuple[TimedNote, ...], time_seconds: float, key_states, pressed_keys) -> None:
    active = [note.key_index for note in notes if note.onset <= time_seconds < note.offset]
    trace["times"].append(float(time_seconds))
    trace["target_vectors"].append(binary_key_vector(active))
    trace["pressed_vectors"].append(binary_key_vector(pressed_keys))
    trace["key_states"].append(np.asarray(key_states, dtype=np.float32).copy())


def _max_target_state(notes: tuple[TimedNote, ...], key_states: np.ndarray) -> float:
    keys = sorted({note.key_index for note in notes})
    if not keys or key_states.size == 0:
        return 0.0
    return float(np.max(key_states[:, keys]))


def _max_unintended_state(notes: tuple[TimedNote, ...], times: list[float], key_states: np.ndarray) -> float:
    return max(_per_step_unintended_states(notes, times, key_states), default=0.0)


def _integrated_unintended_state(notes: tuple[TimedNote, ...], times: list[float], key_states: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    dt = float(np.median(np.diff(np.asarray(times, dtype=np.float64))))
    return float(sum(_per_step_unintended_states(notes, times, key_states)) * dt)


def _per_step_unintended_states(notes: tuple[TimedNote, ...], times: list[float], key_states: np.ndarray) -> list[float]:
    values = []
    for row, time_seconds in enumerate(times):
        states = key_states[row].copy()
        for note in notes:
            if note.onset <= time_seconds < note.offset:
                states[note.key_index] = 0.0
        values.append(float(np.max(states)))
    return values


def _build_transcriber(name: str):
    if name == "basic_pitch":
        return BasicPitchTranscriber(
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length_ms=80.0,
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
    *,
    onset_tolerance: float,
    offset_tolerance: float | None,
) -> IndirectPipelineSymbolicResult:
    sequence = timed_notes_to_controller_sequence(
        output.notes,
        midi_min=config.midi_min,
        midi_max=config.midi_max,
        confidence_threshold=config.confidence_threshold,
        range_policy="drop",
        duplicate_policy=config.duplicate_policy,
        allow_polyphony=True,
    )
    metrics = transcription_note_metrics(
        expected_notes,
        sequence.notes,
        onset_tolerance_seconds=onset_tolerance,
        offset_tolerance_seconds=offset_tolerance,
    )
    return IndirectPipelineSymbolicResult(output, sequence, metrics)


def _empty_sequence_row(item: RenderedBenchmarkItem, model_label: str, outcome: str) -> dict[str, Any]:
    return {
        "model_label": model_label,
        "pipeline": "pipeline1",
        "sequence_name": item.sequence_name,
        "sequence": "-".join(str(pitch) for pitch in item.pitches),
        "sequence_length": len(item.pitches),
        "archetype": _archetype_from_name(item.sequence_name),
        "audio_mode": "basic_pitch",
        "target_event_count": len(item.pitches),
        "target_event_hit_count": 0,
        "target_event_hit_rate": 0.0,
        "pressed_key_precision": 0.0,
        "pressed_key_recall": 0.0,
        "pressed_key_f1": 0.0,
        "timestep_precision": 0.0,
        "timestep_recall": 0.0,
        "timestep_f1": 0.0,
        "max_target_key_state": 0.0,
        "max_unintended_key_state": 0.0,
        "integrated_unintended_key_state": 0.0,
        "strict_outcome": outcome,
    }


def _parse_sequence_specs(specs: Iterable[str]) -> tuple[tuple[int, ...], ...]:
    parsed = []
    for spec in specs:
        try:
            sequence = tuple(int(value.strip()) for value in spec.split(",") if value.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid MIDI sequence specification: {spec!r}") from exc
        if not sequence:
            raise ValueError("Audio intervention sequences must not be empty.")
        parsed.append(sequence)
    return tuple(parsed)


def _audio_modes_by_sequence(
    sequences,
    *,
    include_audio_interventions: bool,
    intervention_sequences: tuple[tuple[int, ...], ...] = (),
) -> dict[tuple[int, ...], tuple[str, ...]]:
    if not include_audio_interventions:
        return {tuple(sequence): ("correct",) for sequence in sequences}
    available = {tuple(sequence) for sequence in sequences}
    requested = set(intervention_sequences)
    unknown = requested - available
    if unknown:
        raise ValueError(f"Audio intervention sequences are absent from the benchmark: {sorted(unknown)}")
    if not requested:
        representative_by_length = {}
        for sequence in sequences:
            representative_by_length.setdefault(len(sequence), tuple(sequence))
        requested = set(representative_by_length.values())
    return {
        tuple(sequence): ("correct", "zero", "mismatched") if tuple(sequence) in requested else ("correct",)
        for sequence in sequences
    }


def _mismatch_sequence(sequence: tuple[int, ...], sequences: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    same_length = [tuple(candidate) for candidate in sequences if len(candidate) == len(sequence) and tuple(candidate) != tuple(sequence)]
    candidates = same_length or [tuple(candidate) for candidate in sequences if tuple(candidate) != tuple(sequence)]
    if not candidates:
        return tuple(sequence)
    return candidates[0]


def _action_dependence_row(agent, env, sequence, clip_index, mismatch_sequence, mismatch_clip_id, model_label):
    obs, _ = env.reset_to_clip_index(clip_index, seed=17)
    correct = agent.act(env.observation_for_audio_mode(obs, mode="correct"), deterministic=True)
    zero = agent.act(env.observation_for_audio_mode(obs, mode="zero"), deterministic=True)
    mismatch = agent.act(
        env.observation_for_audio_mode(obs, mode="mismatched", mismatched_clip_id=mismatch_clip_id),
        deterministic=True,
    )
    return {
        "model_label": _pipeline2_label(model_label),
        "sequence": "-".join(str(pitch) for pitch in sequence),
        "sequence_length": len(sequence),
        "mismatched_sequence": "-".join(str(pitch) for pitch in mismatch_sequence),
        "mean_abs_action_diff_correct_zero": float(np.mean(np.abs(correct - zero))),
        "max_abs_action_diff_correct_zero": float(np.max(np.abs(correct - zero))),
        "mean_abs_action_diff_correct_mismatched": float(np.mean(np.abs(correct - mismatch))),
        "max_abs_action_diff_correct_mismatched": float(np.max(np.abs(correct - mismatch))),
    }


def _aggregate_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    metrics = [
        "pressed_key_precision",
        "pressed_key_recall",
        "pressed_key_f1",
        "timestep_precision",
        "timestep_recall",
        "timestep_f1",
        "target_event_hit_rate",
        "ordered_event_accuracy",
        "transition_event_pair_accuracy",
        "max_unintended_key_state",
        "integrated_unintended_key_state",
        "unintended_presses_per_target_event",
        "correctly_executed_prefix_length",
    ]
    out = []
    for group, subset in sorted(groups.items()):
        row = {key: value for key, value in zip(keys, group)}
        row["sequence_count"] = len(subset)
        for metric in metrics:
            values = [float(item[metric]) for item in subset if item.get(metric) is not None]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_min"] = float(np.min(values))
                row[f"{metric}_max"] = float(np.max(values))
        out.append(row)
    return out


def _fairness_audit(*, max_horizon: int, benchmark) -> dict[str, Any]:
    return {
        "pipeline2_waveform_context_seconds": {"past": 0.10, "future": 0.40},
        "pipeline1_symbolic_lookahead": 1,
        "pipeline2_robo_goal_lookahead": 1,
        "sequence_timing_profile": "aligned",
        "note_duration": benchmark.timing.note_duration,
        "note_gap": benchmark.timing.note_gap,
        "control_timestep_seconds": 0.05,
        "episode_horizon_steps": int(max_horizon),
        "episode_horizon_policy": "ceil(last_note_offset + 0.80 seconds) / 0.05, shared across selected sequences",
        "reset_behavior": "fresh environment reset per model/sequence/audio-mode evaluation",
        "timing_limitation": "Pipeline 2 has 0.40 s audio future context; symbolic Pipeline 1 uses the frozen controller's native one-step goal lookahead.",
    }


def _sequence_payload(sequence, timing) -> dict[str, Any]:
    notes = sequence_notes(sequence, timing)
    return {
        "name": sequence.name,
        "archetype": sequence.archetype,
        "length": sequence.length,
        "midi_pitches": list(sequence.pitches),
        "key_indices": list(sequence.key_indices),
        "onset_times": [note.onset for note in notes],
        "offset_times": [note.offset for note in notes],
        "durations": [note.duration for note in notes],
    }


def _pipeline2_label(label: str) -> str:
    return {
        "pipeline2_seed13_1m": "Pipeline2 seed13 1M",
        "pipeline2_seed61_1m": "Pipeline2 seed61 1M",
        "pipeline2_seed13_1p25m": "Pipeline2 seed13 1.25M",
    }.get(label, label)


def _archetype_from_name(sequence_name: str) -> str:
    parts = sequence_name.split("_")
    if "adjacent" in parts:
        return "adjacent_scalar_walk"
    if "reversed" in parts:
        return "reversed_oscillating_walk"
    if "nonadjacent" in parts:
        return "non_adjacent_jumps"
    if "repeated" in parts:
        return "repeated_motif"
    if "seeded" in parts:
        return "seeded_balanced_sequence"
    return "unknown"


def _control_timestep(env: GeneralOneHandGoalEnv) -> float:
    try:
        return float(env.env.control_timestep())
    except Exception:
        return 0.05


def _json_safe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        safe = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                safe[key] = json.dumps(value, sort_keys=True)
            elif isinstance(value, np.generic):
                safe[key] = value.item()
            else:
                safe[key] = value
        out.append(safe)
    return out


if __name__ == "__main__":
    main()
