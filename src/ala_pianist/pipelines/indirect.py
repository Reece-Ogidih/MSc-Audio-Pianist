"""Interfaces for the indirect audio-to-MIDI-to-action pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np

from ala_pianist.audio.transcriber import AudioToMidiTranscriber, TranscriptionOutput
from ala_pianist.evaluation.transcription_metrics import (
    TranscriptionMetrics,
    transcription_note_metrics,
)
from ala_pianist.music import (
    SequenceTimingConfig,
    generate_sequence_events,
    sequence_timing_from_profile,
    write_sequence_midi,
)
from ala_pianist.music.timed_notes import (
    ControllerGoalSequence,
    TimedNote,
    note_event_to_timed_note,
    timed_notes_to_controller_sequence,
    write_controller_goal_midi,
)


SUPPORTED_RIGHT_HAND_PITCHES = tuple(range(72, 77))

BENCHMARK_SEQUENCE_PITCHES: tuple[tuple[int, ...], ...] = (
    (72,),
    (73,),
    (74,),
    (75,),
    (76,),
    (72, 73),
    (73, 72),
    (73, 74),
    (74, 73),
    (74, 75),
    (75, 74),
    (75, 76),
    (76, 75),
)


@dataclass(frozen=True)
class IndirectPipelineConfig:
    midi_min: int = 72
    midi_max: int = 76
    confidence_threshold: float = 0.0
    range_policy: str = "error"
    duplicate_policy: str = "merge"
    sequence_timing_profile: str = "aligned"


@dataclass(frozen=True)
class IndirectPipelineInput:
    audio_path: Path
    oracle_midi_path: Path | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class IndirectPipelineSymbolicResult:
    transcription: TranscriptionOutput
    controller_sequence: ControllerGoalSequence
    transcription_metrics: TranscriptionMetrics | None = None


@dataclass(frozen=True)
class RenderedBenchmarkItem:
    sequence_name: str
    pitches: tuple[int, ...]
    midi_path: Path
    wav_path: Path
    notes: tuple[TimedNote, ...]


def benchmark_sequence_name(pitches: Iterable[int]) -> str:
    values = tuple(int(pitch) for pitch in pitches)
    if len(values) == 1:
        return f"anchor_{values[0]}"
    return "transition_" + "_".join(str(pitch) for pitch in values)


def benchmark_sequence_events(
    pitches: Iterable[int],
    *,
    timing: SequenceTimingConfig | None = None,
) -> tuple:
    """Return preserved-timing MIDI events for a benchmark pitch sequence."""

    return generate_sequence_events(
        tuple(int(pitch) for pitch in pitches),
        midi_min=min(SUPPORTED_RIGHT_HAND_PITCHES),
        midi_max=max(SUPPORTED_RIGHT_HAND_PITCHES),
        timing=timing or sequence_timing_from_profile("aligned"),
    )


def _explicit_soundfont_from_env() -> Path | None:
    for env_name in ("SOUNDFONT", "ALA_PIANIST_SOUNDFONT"):
        value = os.environ.get(env_name)
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{env_name} is set to {value!r}, but that path does not exist or is not a file. "
                "Unset the variable to use default soundfont discovery."
            )
        return path
    return None


def _default_soundfont_candidates() -> list[Path]:
    project_root = Path(__file__).resolve().parents[3]
    env_project_root = os.environ.get("PROJECT_ROOT")
    roots = [project_root]
    if env_project_root:
        roots.insert(0, Path(env_project_root).expanduser().resolve())

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2",
                root / "third_party/robopianist/third_party/soundfonts/TimGM6mb.sf2",
            ]
        )
    candidates.extend(
        [
            Path("/home/reece_dev/msc-audio-pianist/third_party/robopianist/robopianist/soundfonts/TimGM6mb.sf2"),
            Path("/home/reece_dev/msc-audio-pianist/third_party/robopianist/third_party/soundfonts/TimGM6mb.sf2"),
            Path("/home/reece_dev/miniforge3/envs/pianist/lib/python3.10/site-packages/pretty_midi/TimGM6mb.sf2"),
        ]
    )
    return candidates


def find_default_soundfont() -> Path:
    """Locate the existing RoboPianist TimGM6mb soundfont."""

    explicit = _explicit_soundfont_from_env()
    if explicit is not None:
        return explicit

    candidates = _default_soundfont_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    candidate_text = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate TimGM6mb.sf2 in known project/environment paths. "
        "Set SOUNDFONT to an explicit .sf2 path if it lives elsewhere.\n"
        f"Checked:\n{candidate_text}"
    )


def render_midi_with_fluidsynth(
    midi_path: str | Path,
    wav_path: str | Path,
    *,
    soundfont_path: str | Path,
    sample_rate: int = 44100,
    gain: float = 0.5,
) -> Path:
    """Render MIDI to WAV with FluidSynth in non-interactive mode."""

    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "fluidsynth",
        "-ni",
        "-g",
        str(float(gain)),
        "-r",
        str(int(sample_rate)),
        str(soundfont_path),
        str(midi_path),
        "-F",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return wav_path


def create_rendered_benchmark(
    output_dir: str | Path,
    *,
    soundfont_path: str | Path | None = None,
    sample_rate: int = 44100,
    gain: float = 0.5,
    timing: SequenceTimingConfig | None = None,
) -> tuple[RenderedBenchmarkItem, ...]:
    """Generate the 13 MIDI/WAV benchmark clips and write a manifest."""

    root = Path(output_dir)
    midi_dir = root / "midi"
    wav_dir = root / "wav"
    soundfont = Path(soundfont_path) if soundfont_path is not None else find_default_soundfont()
    timing = timing or sequence_timing_from_profile("aligned")
    items = []
    for pitches in BENCHMARK_SEQUENCE_PITCHES:
        name = benchmark_sequence_name(pitches)
        midi_path = write_benchmark_sequence_midi(pitches, midi_dir / f"{name}.mid", timing=timing)
        wav_path = render_midi_with_fluidsynth(
            midi_path,
            wav_dir / f"{name}.wav",
            soundfont_path=soundfont,
            sample_rate=sample_rate,
            gain=gain,
        )
        notes = note_events_to_timed_notes(benchmark_sequence_events(pitches, timing=timing))
        items.append(RenderedBenchmarkItem(name, pitches, midi_path, wav_path, notes))
    write_benchmark_manifest(
        items,
        root / "benchmark_manifest.json",
        soundfont_path=soundfont,
        sample_rate=sample_rate,
        gain=gain,
        timing=timing,
    )
    return tuple(items)


def write_benchmark_manifest(
    items: Iterable[RenderedBenchmarkItem],
    path: str | Path,
    *,
    soundfont_path: str | Path,
    sample_rate: int,
    gain: float,
    timing: SequenceTimingConfig,
) -> Path:
    """Write benchmark metadata beside rendered audio."""

    try:
        fluidsynth_version = subprocess.run(
            ["fluidsynth", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.splitlines()[0]
    except Exception:
        fluidsynth_version = "unknown"
    payload = {
        "benchmark_name": "five_note_rendered_benchmark_v1",
        "midi_min": min(SUPPORTED_RIGHT_HAND_PITCHES),
        "midi_max": max(SUPPORTED_RIGHT_HAND_PITCHES),
        "key_index_mapping": "key_index = midi_pitch - 21",
        "right_hand": True,
        "monophonic_reference": True,
        "sustain": "none",
        "timing": asdict(timing),
        "synthesis": {
            "renderer": "fluidsynth",
            "renderer_version": fluidsynth_version,
            "soundfont": str(soundfont_path),
            "sample_rate": int(sample_rate),
            "gain": float(gain),
        },
        "sequences": [
            {
                "sequence_name": item.sequence_name,
                "midi_pitches": list(item.pitches),
                "key_indices": [pitch - 21 for pitch in item.pitches],
                "midi_path": str(item.midi_path),
                "wav_path": str(item.wav_path),
                "onset_times": [note.onset for note in item.notes],
                "offset_times": [note.offset for note in item.notes],
                "durations": [note.duration for note in item.notes],
            }
            for item in items
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_benchmark_sequence_midi(
    pitches: Iterable[int],
    path: str | Path,
    *,
    timing: SequenceTimingConfig | None = None,
) -> Path:
    """Write one of the current five-note benchmark sequences."""

    return write_sequence_midi(
        tuple(int(pitch) for pitch in pitches),
        path,
        midi_min=min(SUPPORTED_RIGHT_HAND_PITCHES),
        midi_max=max(SUPPORTED_RIGHT_HAND_PITCHES),
        timing=timing or sequence_timing_from_profile("aligned"),
        title="ALA Pianist indirect benchmark sequence",
    )


def run_symbolic_frontend(
    *,
    audio_path: str | Path,
    transcriber: AudioToMidiTranscriber,
    config: IndirectPipelineConfig | None = None,
    expected_notes: Iterable[TimedNote] | None = None,
) -> IndirectPipelineSymbolicResult:
    """Run audio transcription and convert output to controller-compatible timing."""

    config = config or IndirectPipelineConfig()
    transcription = transcriber.transcribe(audio_path)
    controller_sequence = timed_notes_to_controller_sequence(
        transcription.notes,
        midi_min=config.midi_min,
        midi_max=config.midi_max,
        confidence_threshold=config.confidence_threshold,
        range_policy=config.range_policy,  # type: ignore[arg-type]
        duplicate_policy=config.duplicate_policy,  # type: ignore[arg-type]
        allow_polyphony=True,
    )
    metrics = None
    if expected_notes is not None:
        metrics = transcription_note_metrics(tuple(expected_notes), controller_sequence.notes)
    return IndirectPipelineSymbolicResult(
        transcription=transcription,
        controller_sequence=controller_sequence,
        transcription_metrics=metrics,
    )


def note_events_to_timed_notes(events: Iterable) -> tuple[TimedNote, ...]:
    """Convert existing generated MIDI events into canonical timed notes."""

    return tuple(note_event_to_timed_note(event, source="generated_midi") for event in events)


def write_csv_rows(rows: Iterable[dict], path: str | Path) -> Path:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_symbolic_result_summary(
    result: IndirectPipelineSymbolicResult,
    path: str | Path,
    *,
    controller_midi_path: str | Path | None = None,
) -> Path:
    """Save a compact JSON summary without generated checkpoints or media."""

    payload = {
        "transcriber_name": result.transcription.transcriber_name,
        "notes": [
            {
                "pitch": note.pitch,
                "key_index": note.key_index,
                "onset": note.onset,
                "offset": note.offset,
                "duration": note.duration,
                "confidence": note.confidence,
                "source": note.source,
                "metadata": note.metadata,
            }
            for note in result.controller_sequence.notes
        ],
        "controller_midi_path": None if controller_midi_path is None else str(controller_midi_path),
        "transcription_metrics": None
        if result.transcription_metrics is None
        else result.transcription_metrics.as_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_controller_midi_from_result(
    result: IndirectPipelineSymbolicResult,
    path: str | Path,
) -> Path:
    """Write downstream controller MIDI from a symbolic frontend result."""

    return write_controller_goal_midi(result.controller_sequence, path)
