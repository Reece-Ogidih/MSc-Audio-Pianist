"""Simple deterministic monophonic pitch transcription for generated tones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ala_pianist.music import NoteEvent


@dataclass(frozen=True)
class TranscriptionResult:
    """Estimated note events from a generated monophonic WAV."""

    note_events: tuple[NoteEvent, ...]
    frame_pitches: tuple[int | None, ...]


def frequency_to_midi(frequency: float) -> int:
    return int(round(69 + 12 * np.log2(float(frequency) / 440.0)))


def transcribe_monophonic_wav(
    path: str | Path,
    *,
    frame_seconds: float = 0.04,
    hop_seconds: float = 0.02,
    energy_threshold: float = 0.02,
    min_note_duration: float = 0.08,
) -> TranscriptionResult:
    """Transcribe clear generated monophonic tones using windowed FFT peaks."""

    sample_rate, audio = wavfile.read(path)
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if signal.size == 0:
        return TranscriptionResult(note_events=(), frame_pitches=())
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak

    frame_size = max(64, int(round(frame_seconds * sample_rate)))
    hop_size = max(1, int(round(hop_seconds * sample_rate)))
    window = np.hanning(frame_size)
    pitches: list[int | None] = []
    times: list[float] = []
    for start in range(0, max(1, signal.size - frame_size + 1), hop_size):
        frame = signal[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        rms = float(np.sqrt(np.mean(frame**2)))
        if rms < energy_threshold:
            pitch = None
        else:
            spectrum = np.abs(np.fft.rfft(frame * window))
            freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
            valid = (freqs >= 150.0) & (freqs <= 1200.0)
            idxs = np.flatnonzero(valid)
            best_idx = idxs[int(np.argmax(spectrum[valid]))]
            pitch = frequency_to_midi(float(freqs[best_idx]))
        pitches.append(pitch)
        times.append(start / sample_rate)

    events = _pitches_to_events(pitches, times, hop_seconds, min_note_duration)
    return TranscriptionResult(note_events=tuple(events), frame_pitches=tuple(pitches))


def transcription_accuracy(expected: list[NoteEvent], observed: list[NoteEvent] | tuple[NoteEvent, ...]) -> dict:
    expected_pitches = [event.pitch for event in expected]
    observed_pitches = [event.pitch for event in observed]
    matches = sum(1 for left, right in zip(expected_pitches, observed_pitches) if left == right)
    return {
        "expected_count": len(expected_pitches),
        "observed_count": len(observed_pitches),
        "pitch_matches": matches,
        "pitch_accuracy": matches / max(1, len(expected_pitches)),
        "expected_pitches": expected_pitches,
        "observed_pitches": observed_pitches,
    }


def _pitches_to_events(
    pitches: list[int | None],
    times: list[float],
    hop_seconds: float,
    min_note_duration: float,
) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    current_pitch = None
    start_time = None
    last_time = 0.0
    for pitch, time in zip(pitches, times):
        if pitch != current_pitch:
            if current_pitch is not None and start_time is not None:
                duration = max(0.0, last_time + hop_seconds - start_time)
                if duration >= min_note_duration:
                    events.append(NoteEvent(current_pitch, start_time, duration, 90))
            current_pitch = pitch
            start_time = time if pitch is not None else None
        last_time = time
    if current_pitch is not None and start_time is not None:
        duration = max(0.0, last_time + hop_seconds - start_time)
        if duration >= min_note_duration:
            events.append(NoteEvent(current_pitch, start_time, duration, 90))
    return _merge_repeated(events)


def _merge_repeated(events: list[NoteEvent], max_gap: float = 0.06) -> list[NoteEvent]:
    merged: list[NoteEvent] = []
    for event in events:
        if merged and merged[-1].pitch == event.pitch:
            prev = merged[-1]
            prev_end = prev.start + prev.duration
            if event.start - prev_end <= max_gap:
                merged[-1] = NoteEvent(
                    prev.pitch,
                    prev.start,
                    event.start + event.duration - prev.start,
                    max(prev.velocity, event.velocity),
                )
                continue
        merged.append(event)
    return merged
