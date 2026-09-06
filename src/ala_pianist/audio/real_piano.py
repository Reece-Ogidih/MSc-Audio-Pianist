"""Utilities for real-piano distribution-shift evaluation.

The primary workflow uses isolated single-note recordings and constructs the
existing synthetic benchmark schedules from those recordings. This keeps target
MIDI timing fixed while changing only the acoustic source.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from ala_pianist.evaluation.long_horizon import load_long_horizon_benchmark, sequence_notes
from ala_pianist.music.sequence_generation import write_sequence_midi


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".m4a", ".mp3")
RECORDING_RE = re.compile(r"^midi(?P<pitch>\d{2,3})_take(?P<take>\d{1,3})\.(?P<ext>wav|flac|m4a|mp3)$", re.I)
NOTE_NAME_RECORDING_RE = re.compile(
    r"^(?P<note>C#|D#|C|D|E)\s+take\s+(?P<take>\d{1,3})\.(?P<ext>wav|flac|m4a|mp3)$",
    re.I,
)
NOTE_NAME_TO_MIDI = {"C": 72, "C#": 73, "D": 74, "D#": 75, "E": 76}
MIDI_TO_NOTE_NAME = {value: key for key, value in NOTE_NAME_TO_MIDI.items()}
DEFAULT_PITCHES = tuple(range(72, 77))
DEFAULT_TAKES_PER_PITCH = 3
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class RawRecording:
    pitch: int
    take: int
    path: Path
    sha256: str
    note_name: str
    filename_style: str


@dataclass(frozen=True)
class AudioStats:
    sample_rate: int
    channels: int
    duration_seconds: float
    peak_abs: float
    rms: float
    clipped: bool
    mostly_silence: bool
    clipping_fraction: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    estimated_fundamental_hz: float | None
    estimated_midi_pitch: float | None
    estimated_note_name: str | None


def discover_raw_recordings(raw_dir: str | Path) -> tuple[RawRecording, ...]:
    """Discover MIDI-style and experiment note-name recording files."""

    raw_dir = Path(raw_dir)
    recordings: list[RawRecording] = []
    for path in sorted(raw_dir.iterdir() if raw_dir.exists() else ()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        match = RECORDING_RE.match(path.name)
        note_match = NOTE_NAME_RECORDING_RE.match(path.name)
        if match:
            pitch = int(match.group("pitch"))
            take = int(match.group("take"))
            note_name = MIDI_TO_NOTE_NAME.get(pitch, f"midi{pitch}")
            filename_style = "midi"
        elif note_match:
            note_name = _canonical_note_name(note_match.group("note"))
            pitch = NOTE_NAME_TO_MIDI[note_name]
            take = int(note_match.group("take"))
            filename_style = "note_name"
        else:
            continue
        recordings.append(
            RawRecording(
                pitch=pitch,
                take=take,
                path=path,
                sha256=sha256_file(path),
                note_name=note_name,
                filename_style=filename_style,
            )
        )
    return tuple(recordings)


def validate_raw_recordings(
    raw_dir: str | Path,
    *,
    pitches: Iterable[int] = DEFAULT_PITCHES,
    takes_per_pitch: int = DEFAULT_TAKES_PER_PITCH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the raw recording set without modifying files."""

    recordings = discover_raw_recordings(raw_dir)
    rows = []
    by_pitch = {int(pitch): [] for pitch in pitches}
    for recording in recordings:
        if recording.pitch in by_pitch:
            by_pitch[recording.pitch].append(recording)
        status = "ok"
        error = ""
        stats: AudioStats | None = None
        try:
            stats = audio_stats(recording.path)
            if stats.duration_seconds < 0.10:
                status = "too_short"
            elif stats.mostly_silence:
                status = "mostly_silence"
            elif stats.clipped:
                status = "clipping_warning"
        except Exception as exc:
            status = "invalid_audio"
            error = str(exc)
        row = {
            "pitch": recording.pitch,
            "note_name": recording.note_name,
            "take": recording.take,
            "filename_style": recording.filename_style,
            "path": str(recording.path),
            "source_format": recording.path.suffix.lower().lstrip("."),
            "sha256": recording.sha256,
            "status": status,
            "error": error,
        }
        if stats is not None:
            row.update(asdict(stats))
            row["detected_pitch_compatible"] = _detected_pitch_compatible(recording.pitch, stats.estimated_midi_pitch)
        rows.append(row)

    for pitch, pitch_recordings in by_pitch.items():
        takes = [recording.take for recording in pitch_recordings]
        duplicate_takes = sorted({take for take in takes if takes.count(take) > 1})
        if duplicate_takes:
            rows.append(
                {
                    "pitch": pitch,
                    "note_name": MIDI_TO_NOTE_NAME.get(pitch, ""),
                    "take": ",".join(str(take) for take in duplicate_takes),
                    "path": "",
                    "sha256": "",
                    "status": "duplicate_takes",
                    "error": f"Duplicate take identifiers for pitch {pitch}: {duplicate_takes}.",
                }
            )
        if len(pitch_recordings) != takes_per_pitch:
            rows.append(
                {
                    "pitch": pitch,
                    "note_name": MIDI_TO_NOTE_NAME.get(pitch, ""),
                    "take": "",
                    "path": "",
                    "sha256": "",
                    "status": "missing_takes",
                    "error": f"Expected exactly {takes_per_pitch} takes, found {len(pitch_recordings)}.",
                }
            )
    summary = {
        "raw_dir": str(Path(raw_dir)),
        "required_pitches": list(int(p) for p in pitches),
        "takes_per_pitch": int(takes_per_pitch),
        "recording_count": len(recordings),
        "ok_or_warning_count": sum(row["status"] in {"ok", "clipping_warning"} for row in rows),
        "failure_count": sum(row["status"] not in {"ok", "clipping_warning"} for row in rows),
        "ignored_non_audio_sidecars": sorted(
            path.name
            for path in Path(raw_dir).iterdir()
            if path.is_file() and path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS
        )
        if Path(raw_dir).exists()
        else [],
        "ready": all(len(by_pitch[pitch]) == takes_per_pitch for pitch in by_pitch)
        and all(len({recording.take for recording in by_pitch[pitch]}) == takes_per_pitch for pitch in by_pitch)
        and all(row["status"] in {"ok", "clipping_warning"} for row in rows),
    }
    return rows, summary


def preprocess_recordings(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    peak_normalise_to: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert raw clips to mono 16 kHz WAV with conservative trimming."""

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for recording in discover_raw_recordings(raw_dir):
        output = processed_dir / f"midi{recording.pitch}_take{recording.take:02d}.wav"
        waveform, source_rate = load_audio(recording.path)
        before = audio_stats(recording.path)
        trimmed = trim_leading_trailing_silence(waveform, source_rate, preserve_attack_seconds=0.025)
        resampled = resample_linear(trimmed, source_rate, target_sample_rate)
        peak = float(np.max(np.abs(resampled))) if resampled.size else 0.0
        if peak > 0:
            resampled = np.clip(resampled / peak * float(peak_normalise_to), -0.98, 0.98)
        sf.write(output, resampled.astype(np.float32), target_sample_rate)
        after = stats_from_waveform(resampled, target_sample_rate)
        rows.append(
            {
                "pitch": recording.pitch,
                "take": recording.take,
                "source_path": str(recording.path),
                "source_sha256": recording.sha256,
                "source_note_name": recording.note_name,
                "source_take": recording.take,
                "source_filename_style": recording.filename_style,
                "processed_path": str(output),
                "processed_sha256": sha256_file(output),
                "source_sample_rate": source_rate,
                "source_channels": before.channels,
                "source_duration_seconds": before.duration_seconds,
                "source_peak_abs": before.peak_abs,
                "source_rms": before.rms,
                "source_leading_silence_seconds": before.leading_silence_seconds,
                "source_trailing_silence_seconds": before.trailing_silence_seconds,
                "estimated_fundamental_hz": before.estimated_fundamental_hz,
                "estimated_midi_pitch": before.estimated_midi_pitch,
                "processed_sample_rate": target_sample_rate,
                "processed_duration_seconds": after.duration_seconds,
                "processed_peak_abs": after.peak_abs,
                "processed_rms": after.rms,
                "processed_clipped": after.clipped,
            }
        )
    summary = {
        "processed_dir": str(processed_dir),
        "processed_count": len(rows),
        "target_sample_rate": int(target_sample_rate),
        "normalisation": f"peak normalise each trimmed note clip to {peak_normalise_to}",
        "trimming": "conservative silence trim preserving 25 ms before attack where available",
    }
    return rows, summary


def construct_real_audio_benchmarks(
    *,
    processed_dir: str | Path,
    output_dir: str | Path,
    manifests: Iterable[str | Path],
    realizations: int = 3,
    seed: int = 20260823,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Construct benchmark WAVs from processed single-note clips."""

    processed = _processed_by_pitch(Path(processed_dir))
    output_dir = Path(output_dir)
    rows = []
    for manifest in manifests:
        benchmark = load_long_horizon_benchmark(manifest, allow_trained_short=True)
        for sequence in benchmark.sequences:
            notes = sequence_notes(sequence, benchmark.timing)
            for realization in range(1, int(realizations) + 1):
                assignments = []
                total_duration = max(note.offset for note in notes) + 0.8
                mix = np.zeros(int(np.ceil(total_duration * sample_rate)), dtype=np.float32)
                for event_index, note in enumerate(notes):
                    choices = processed[int(note.pitch)]
                    choice = choices[(int(realization) - 1 + event_index) % len(choices)]
                    waveform, sr = sf.read(choice, always_2d=False)
                    waveform = np.asarray(waveform, dtype=np.float32)
                    if waveform.ndim == 2:
                        waveform = waveform.mean(axis=1)
                    if sr != sample_rate:
                        waveform = resample_linear(waveform, int(sr), sample_rate)
                    start = int(round(note.onset * sample_rate))
                    stop = min(mix.size, start + waveform.size)
                    if stop > start:
                        mix[start:stop] += waveform[: stop - start]
                    assignments.append(
                        {
                            "event_index": event_index,
                            "pitch": int(note.pitch),
                            "onset": float(note.onset),
                            "offset": float(note.offset),
                            "take_path": str(choice),
                        }
                    )
                peak = float(np.max(np.abs(mix))) if mix.size else 0.0
                if peak > 0.98:
                    mix = mix / peak * 0.98
                rel_dir = output_dir / benchmark.benchmark_name / sequence.name
                rel_dir.mkdir(parents=True, exist_ok=True)
                wav_path = rel_dir / f"realization_{realization:02d}.wav"
                midi_path = rel_dir / f"realization_{realization:02d}.mid"
                if not midi_path.exists():
                    write_sequence_midi(
                        sequence.pitches,
                        midi_path,
                        midi_min=min(benchmark.midi_pitches),
                        midi_max=max(benchmark.midi_pitches),
                        timing=benchmark.timing,
                        title=f"ALA real-audio target {benchmark.benchmark_name} {sequence.name}",
                    )
                sf.write(wav_path, mix.astype(np.float32), sample_rate)
                rows.append(
                    {
                        "benchmark": benchmark.benchmark_name,
                        "manifest_path": str(manifest),
                        "sequence_name": sequence.name,
                        "sequence": "-".join(str(pitch) for pitch in sequence.pitches),
                        "realization": realization,
                        "midi_path": str(midi_path),
                        "midi_sha256": sha256_file(midi_path),
                        "wav_path": str(wav_path),
                        "wav_sha256": sha256_file(wav_path),
                        "take_assignment_json": json.dumps(assignments, sort_keys=True),
                        "sample_rate": sample_rate,
                        "duration_seconds": mix.size / sample_rate,
                        "peak_abs": float(np.max(np.abs(mix))) if mix.size else 0.0,
                    }
                )
    summary = {
        "output_dir": str(output_dir),
        "realization_count": int(realizations),
        "constructed_clip_count": len(rows),
        "seed": int(seed),
        "sample_rate": int(sample_rate),
    }
    return rows, summary


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        waveform, sample_rate = sf.read(path, always_2d=False)
    else:
        waveform, sample_rate = _load_audio_with_ffmpeg(path)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return waveform.reshape(-1), int(sample_rate)


def audio_stats(path: str | Path) -> AudioStats:
    path = Path(path)
    if path.suffix.lower() == ".wav":
        waveform, sample_rate = sf.read(path, always_2d=True)
        return stats_from_waveform(np.asarray(waveform, dtype=np.float32), int(sample_rate))
    waveform, sample_rate = load_audio(path)
    return stats_from_waveform(waveform, sample_rate)


def stats_from_waveform(waveform: np.ndarray, sample_rate: int) -> AudioStats:
    waveform = np.asarray(waveform, dtype=np.float32)
    mono = waveform.mean(axis=1) if waveform.ndim == 2 else waveform.reshape(-1)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
    silence = _silence_edges(mono, sample_rate)
    fundamental = estimate_fundamental_hz(mono, sample_rate)
    estimated_midi = None if fundamental is None else 69.0 + 12.0 * math.log2(float(fundamental) / 440.0)
    return AudioStats(
        sample_rate=int(sample_rate),
        channels=1 if waveform.ndim == 1 else int(waveform.shape[1]),
        duration_seconds=float(mono.size / sample_rate) if sample_rate else 0.0,
        peak_abs=peak,
        rms=rms,
        clipped=bool(peak >= 0.999),
        mostly_silence=bool(rms < 1e-4),
        clipping_fraction=float(np.mean(np.abs(waveform) >= 0.999)) if waveform.size else 0.0,
        leading_silence_seconds=silence[0],
        trailing_silence_seconds=silence[1],
        estimated_fundamental_hz=fundamental,
        estimated_midi_pitch=estimated_midi,
        estimated_note_name=None if estimated_midi is None else _midi_to_note_name(int(round(estimated_midi))),
    )


def trim_leading_trailing_silence(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    threshold_ratio: float = 0.02,
    preserve_attack_seconds: float = 0.025,
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        return waveform
    peak = float(np.max(np.abs(waveform)))
    if peak <= 0:
        return waveform
    mask = np.abs(waveform) >= peak * float(threshold_ratio)
    if not np.any(mask):
        return waveform
    indices = np.flatnonzero(mask)
    pad = int(round(float(preserve_attack_seconds) * int(sample_rate)))
    start = max(0, int(indices[0]) - pad)
    end = min(waveform.size, int(indices[-1]) + pad + 1)
    return waveform[start:end]


def resample_linear(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    duration = waveform.size / float(source_rate)
    count = max(1, int(round(duration * target_rate)))
    source_t = np.linspace(0.0, duration, waveform.size, endpoint=False)
    target_t = np.linspace(0.0, duration, count, endpoint=False)
    return np.interp(target_t, source_t, waveform).astype(np.float32)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_audio_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-",
    ]
    proc = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    return np.frombuffer(proc.stdout, dtype=np.float32), TARGET_SAMPLE_RATE


def _processed_by_pitch(processed_dir: Path) -> dict[int, list[Path]]:
    result: dict[int, list[Path]] = {}
    for recording in discover_raw_recordings(processed_dir):
        result.setdefault(recording.pitch, []).append(recording.path)
    missing = [pitch for pitch in DEFAULT_PITCHES if not result.get(pitch)]
    if missing:
        raise FileNotFoundError(f"Processed recordings missing pitches: {missing}")
    return {pitch: sorted(paths) for pitch, paths in result.items()}


def estimate_fundamental_hz(waveform: np.ndarray, sample_rate: int) -> float | None:
    """Estimate monophonic pitch with a simple bounded autocorrelation."""

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size < sample_rate * 0.05 or sample_rate <= 0:
        return None
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak <= 1e-5:
        return None
    threshold = max(peak * 0.05, 1e-4)
    active = np.flatnonzero(np.abs(waveform) >= threshold)
    if active.size < sample_rate * 0.04:
        return None
    start = int(active[0])
    segment = waveform[start : start + min(int(sample_rate * 0.60), waveform.size - start)]
    if segment.size < sample_rate * 0.04:
        return None
    segment = segment - float(np.mean(segment))
    window = np.hanning(segment.size).astype(np.float32)
    segment = segment * window
    min_lag = max(1, int(sample_rate / 900.0))
    max_lag = min(segment.size - 1, int(sample_rate / 450.0))
    if max_lag <= min_lag:
        return None
    corr = np.correlate(segment, segment, mode="full")[segment.size - 1 :]
    corr[:min_lag] = 0.0
    lag = int(np.argmax(corr[min_lag : max_lag + 1]) + min_lag)
    if corr[lag] <= 0:
        return None
    return float(sample_rate / lag)


def _silence_edges(waveform: np.ndarray, sample_rate: int) -> tuple[float, float]:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0 or sample_rate <= 0:
        return 0.0, 0.0
    peak = float(np.max(np.abs(waveform)))
    if peak <= 0.0:
        return waveform.size / sample_rate, waveform.size / sample_rate
    threshold = max(peak * 0.02, 1e-4)
    active = np.flatnonzero(np.abs(waveform) >= threshold)
    if active.size == 0:
        duration = waveform.size / sample_rate
        return duration, duration
    return float(active[0] / sample_rate), float((waveform.size - active[-1] - 1) / sample_rate)


def _canonical_note_name(value: str) -> str:
    return value.strip().upper().replace("♯", "#")


def _midi_to_note_name(pitch: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    octave = int(pitch) // 12 - 1
    return f"{names[int(pitch) % 12]}{octave}"


def _detected_pitch_compatible(expected_pitch: int, detected_pitch: float | None) -> bool | None:
    if detected_pitch is None:
        return None
    return bool(abs(float(detected_pitch) - float(expected_pitch)) <= 0.75)
