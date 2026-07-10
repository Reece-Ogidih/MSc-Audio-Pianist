"""Deterministic monophonic audio synthesis for generated Pipeline 1 clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ala_pianist.music import NoteEvent


@dataclass(frozen=True)
class SynthesizedClip:
    """Generated WAV clip and its known note events."""

    path: Path
    sample_rate: int
    note_events: tuple[NoteEvent, ...]


def midi_to_frequency(pitch: int) -> float:
    return 440.0 * (2.0 ** ((int(pitch) - 69) / 12.0))


def synthesize_monophonic_wav(
    note_events: list[NoteEvent] | tuple[NoteEvent, ...],
    path: str | Path,
    *,
    sample_rate: int = 16_000,
    amplitude: float = 0.35,
) -> SynthesizedClip:
    """Write a simple harmonic monophonic WAV for deterministic tests."""

    if not note_events:
        raise ValueError("At least one note event is required.")
    total_time = max(event.start + event.duration for event in note_events) + 0.05
    samples = np.zeros(int(np.ceil(total_time * sample_rate)), dtype=np.float64)

    for event in note_events:
        start = int(round(event.start * sample_rate))
        stop = min(samples.size, int(round((event.start + event.duration) * sample_rate)))
        if stop <= start:
            raise ValueError("Note event duration is too short for sample rate.")
        t = np.arange(stop - start, dtype=np.float64) / sample_rate
        freq = midi_to_frequency(event.pitch)
        tone = (
            np.sin(2.0 * np.pi * freq * t)
            + 0.35 * np.sin(2.0 * np.pi * 2.0 * freq * t)
            + 0.15 * np.sin(2.0 * np.pi * 3.0 * freq * t)
        )
        envelope = _fade_envelope(t.size, sample_rate)
        samples[start:stop] += amplitude * envelope * tone

    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples / peak * 0.8
    pcm = np.asarray(np.clip(samples, -1.0, 1.0) * 32767, dtype=np.int16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, pcm)
    return SynthesizedClip(path=path, sample_rate=sample_rate, note_events=tuple(note_events))


def _fade_envelope(length: int, sample_rate: int) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float64)
    fade = min(length // 2, int(0.015 * sample_rate))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=True)
        envelope[:fade] = ramp
        envelope[-fade:] = ramp[::-1]
    return envelope
