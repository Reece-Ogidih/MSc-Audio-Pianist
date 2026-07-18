"""Shared deterministic sequence timing for training and evaluation MIDI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ala_pianist.music.midi_utils import NoteEvent, write_monophonic_midi


@dataclass(frozen=True)
class SequenceTimingConfig:
    """Timing used to generate short monophonic sequence clips."""

    note_duration: float = 0.28
    note_gap: float = 0.12
    velocity: int = 90
    timing_jitter: float = 0.0

    @property
    def step_seconds(self) -> float:
        return self.note_duration + self.note_gap


def sequence_timing_from_profile(
    profile: str,
    *,
    note_duration: float | None = None,
    note_gap: float | None = None,
    velocity: int | None = None,
    timing_jitter: float = 0.0,
) -> SequenceTimingConfig:
    """Return a named deterministic timing profile."""

    if profile == "aligned":
        config = SequenceTimingConfig(note_duration=0.28, note_gap=0.12, velocity=90)
    elif profile == "legacy_curriculum":
        config = SequenceTimingConfig(note_duration=0.45, note_gap=0.05, velocity=90)
    else:
        raise ValueError("sequence timing profile must be 'aligned' or 'legacy_curriculum'.")
    return SequenceTimingConfig(
        note_duration=config.note_duration if note_duration is None else float(note_duration),
        note_gap=config.note_gap if note_gap is None else float(note_gap),
        velocity=config.velocity if velocity is None else int(velocity),
        timing_jitter=float(timing_jitter),
    )


def generate_sequence_events(
    pitches: list[int] | tuple[int, ...],
    *,
    midi_min: int | None = None,
    midi_max: int | None = None,
    timing: SequenceTimingConfig | None = None,
    fingering_fn=None,
) -> tuple[NoteEvent, ...]:
    """Generate deterministic monophonic note events for a pitch sequence."""

    if not pitches:
        raise ValueError("At least one pitch is required.")
    timing = timing or SequenceTimingConfig()
    if timing.note_duration <= 0.0:
        raise ValueError("note_duration must be positive.")
    if timing.note_gap < 0.0:
        raise ValueError("note_gap must be non-negative.")
    if timing.timing_jitter != 0.0:
        raise ValueError("Only deterministic timing_jitter=0.0 is currently supported.")
    midi_min = min(pitches) if midi_min is None else int(midi_min)
    midi_max = max(pitches) if midi_max is None else int(midi_max)
    events = []
    for index, pitch in enumerate(pitches):
        fingering = None if fingering_fn is None else fingering_fn(int(pitch), midi_min, midi_max)
        events.append(
            NoteEvent(
                pitch=int(pitch),
                start=timing.step_seconds * index,
                duration=timing.note_duration,
                velocity=timing.velocity,
                fingering=fingering,
            )
        )
    return tuple(events)


def write_sequence_midi(
    pitches: list[int] | tuple[int, ...],
    path: str | Path,
    *,
    midi_min: int | None = None,
    midi_max: int | None = None,
    timing: SequenceTimingConfig | None = None,
    fingering_fn=None,
    title: str = "ALA Pianist sequence",
) -> Path:
    """Write a deterministic sequence MIDI clip using shared timing."""

    events = generate_sequence_events(
        pitches,
        midi_min=midi_min,
        midi_max=midi_max,
        timing=timing,
        fingering_fn=fingering_fn,
    )
    return write_monophonic_midi(events, path, title=title)


def note_windows(
    pitches: list[int] | tuple[int, ...],
    *,
    timing: SequenceTimingConfig | None = None,
) -> tuple[dict[str, float | int], ...]:
    """Return expected note onset/offset windows for a sequence."""

    timing = timing or SequenceTimingConfig()
    return tuple(
        {
            "pitch": int(pitch),
            "key_index": int(pitch) - 21,
            "start_seconds": timing.step_seconds * index,
            "end_seconds": timing.step_seconds * index + timing.note_duration,
            "duration": timing.note_duration,
        }
        for index, pitch in enumerate(pitches)
    )
