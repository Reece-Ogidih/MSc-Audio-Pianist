"""Utilities for real-piano distribution-shift evaluation.

The primary workflow uses isolated single-note recordings and constructs the
existing synthetic benchmark schedules from those recordings. This keeps target
MIDI timing fixed while changing only the acoustic source.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
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
DEFAULT_PITCHES = tuple(range(72, 77))
DEFAULT_TAKES_PER_PITCH = 5
TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class RawRecording:
    pitch: int
    take: int
    path: Path
    sha256: str


@dataclass(frozen=True)
class AudioStats:
    sample_rate: int
    channels: int
    duration_seconds: float
    peak_abs: float
    rms: float
    clipped: bool
    mostly_silence: bool


def discover_raw_recordings(raw_dir: str | Path) -> tuple[RawRecording, ...]:
    """Discover files named like ``midi72_take01.wav``."""

    raw_dir = Path(raw_dir)
    recordings: list[RawRecording] = []
    for path in sorted(raw_dir.iterdir() if raw_dir.exists() else ()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        match = RECORDING_RE.match(path.name)
        if not match:
            continue
        recordings.append(
            RawRecording(
                pitch=int(match.group("pitch")),
                take=int(match.group("take")),
                path=path,
                sha256=sha256_file(path),
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
            "take": recording.take,
            "path": str(recording.path),
            "sha256": recording.sha256,
            "status": status,
            "error": error,
        }
        if stats is not None:
            row.update(asdict(stats))
        rows.append(row)

    for pitch, pitch_recordings in by_pitch.items():
        if len(pitch_recordings) < takes_per_pitch:
            rows.append(
                {
                    "pitch": pitch,
                    "take": "",
                    "path": "",
                    "sha256": "",
                    "status": "missing_takes",
                    "error": f"Expected at least {takes_per_pitch} takes, found {len(pitch_recordings)}.",
                }
            )
    summary = {
        "raw_dir": str(Path(raw_dir)),
        "required_pitches": list(int(p) for p in pitches),
        "takes_per_pitch": int(takes_per_pitch),
        "recording_count": len(recordings),
        "ok_or_warning_count": sum(row["status"] in {"ok", "clipping_warning"} for row in rows),
        "failure_count": sum(row["status"] not in {"ok", "clipping_warning"} for row in rows),
        "ready": all(len(by_pitch[pitch]) >= takes_per_pitch for pitch in by_pitch)
        and all(row["status"] in {"ok", "clipping_warning"} for row in rows if row.get("path")),
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
        before = stats_from_waveform(waveform, source_rate)
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
                "processed_path": str(output),
                "processed_sha256": sha256_file(output),
                "source_sample_rate": source_rate,
                "source_duration_seconds": before.duration_seconds,
                "source_peak_abs": before.peak_abs,
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
    rng = np.random.default_rng(int(seed))
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
                    choice = choices[int(rng.integers(0, len(choices)))]
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
    waveform, sample_rate = load_audio(path)
    return stats_from_waveform(waveform, sample_rate)


def stats_from_waveform(waveform: np.ndarray, sample_rate: int) -> AudioStats:
    waveform = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
    return AudioStats(
        sample_rate=int(sample_rate),
        channels=1 if waveform.ndim == 1 else int(waveform.shape[1]),
        duration_seconds=float(waveform.reshape(-1).size / sample_rate) if sample_rate else 0.0,
        peak_abs=peak,
        rms=rms,
        clipped=bool(peak >= 0.999),
        mostly_silence=bool(rms < 1e-4),
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
