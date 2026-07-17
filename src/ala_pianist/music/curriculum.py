"""Generated monophonic MIDI curricula for one-hand debug learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ala_pianist.music.midi_utils import NoteEvent, write_monophonic_midi


CURRICULUM_MODES = (
    "single_notes",
    "repeated_notes",
    "two_note_transitions",
    "short_phrases",
    "sequence_cleanup",
)


@dataclass(frozen=True)
class CurriculumClip:
    """Generated clip metadata used by the learning/debug harness."""

    events: tuple[NoteEvent, ...]
    midi_path: Path
    mode: str
    midi_min: int
    midi_max: int

    @property
    def pitches(self) -> tuple[int, ...]:
        return tuple(event.pitch for event in self.events)

    @property
    def key_indices(self) -> tuple[int, ...]:
        return tuple(event.pitch - 21 for event in self.events)

    @property
    def fingerings(self) -> tuple[int | None, ...]:
        return tuple(event.fingering for event in self.events)


def assign_right_hand_fingering(pitch: int, midi_min: int, midi_max: int) -> int:
    """Assign a deterministic approximate right-hand finger label in `[0, 4]`."""

    if midi_max < midi_min:
        raise ValueError("midi_max must be greater than or equal to midi_min.")
    if not midi_min <= pitch <= midi_max:
        raise ValueError(f"Pitch {pitch} is outside [{midi_min}, {midi_max}].")
    span = max(1, midi_max - midi_min)
    relative = (pitch - midi_min) / span
    return int(np.clip(round(relative * 4), 0, 4))


def generate_curriculum_events(
    *,
    mode: str,
    midi_min: int,
    midi_max: int,
    midi_pitches: tuple[int, ...] | list[int] | None = None,
    sequence_pitches: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None = None,
    seed: int = 0,
    clip_index: int = 0,
    note_count: int = 4,
    note_duration: float = 0.45,
    gap: float = 0.05,
    velocity: int = 90,
) -> tuple[NoteEvent, ...]:
    """Generate a tiny deterministic monophonic clip for local one-hand training."""

    if mode not in CURRICULUM_MODES:
        raise ValueError(f"Unknown curriculum mode {mode!r}.")
    sequences = _normalise_sequence_pitches(sequence_pitches)
    pitches = _pitch_array(
        midi_min=midi_min,
        midi_max=midi_max,
        midi_pitches=midi_pitches,
    )
    rng = np.random.default_rng(int(seed) + 9973 * int(clip_index))
    note_count = max(1, int(note_count))

    if mode == "single_notes":
        selected = [int(pitches[clip_index % pitches.size])]
    elif mode == "repeated_notes":
        pitch = int(pitches[clip_index % pitches.size])
        selected = [pitch] * min(note_count, 4)
    elif mode == "two_note_transitions":
        first = int(pitches[clip_index % pitches.size])
        if pitches.size == 1:
            second = first
        else:
            offsets = np.asarray([-2, -1, 1, 2], dtype=int)
            candidates = [first + int(offset) for offset in offsets]
            allowed = set(int(pitch) for pitch in pitches)
            candidates = [pitch for pitch in candidates if pitch in allowed]
            if not candidates:
                candidates = [int(pitch) for pitch in pitches if int(pitch) != first]
            second = int(candidates[clip_index % len(candidates)])
        selected = [first, second]
    elif mode == "sequence_cleanup":
        if sequences is None:
            sequences = ((73,), (75,), (73, 75), (75, 73))
        selected = [int(pitch) for pitch in sequences[clip_index % len(sequences)]]
    else:
        selected = [int(rng.choice(pitches)) for _ in range(min(note_count, 6))]

    events = []
    cursor = 0.0
    for pitch in selected:
        events.append(
            NoteEvent(
                pitch=int(pitch),
                start=cursor,
                duration=float(note_duration),
                velocity=int(velocity),
                fingering=assign_right_hand_fingering(int(pitch), midi_min, midi_max),
            )
        )
        cursor += float(note_duration) + float(gap)
    return tuple(events)


def write_curriculum_midi(
    path: str | Path,
    *,
    mode: str,
    midi_min: int,
    midi_max: int,
    midi_pitches: tuple[int, ...] | list[int] | None = None,
    sequence_pitches: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None = None,
    seed: int = 0,
    clip_index: int = 0,
    note_count: int = 4,
) -> CurriculumClip:
    """Generate and write a curriculum MIDI clip with deterministic fingering labels."""

    events = generate_curriculum_events(
        mode=mode,
        midi_min=midi_min,
        midi_max=midi_max,
        midi_pitches=midi_pitches,
        sequence_pitches=sequence_pitches,
        seed=seed,
        clip_index=clip_index,
        note_count=note_count,
    )
    midi_path = write_monophonic_midi(
        events,
        path,
        title=f"ALA Pianist {mode} {midi_min}-{midi_max}",
    )
    return CurriculumClip(
        events=events,
        midi_path=midi_path,
        mode=mode,
        midi_min=int(midi_min),
        midi_max=int(midi_max),
    )


def _pitch_array(
    *,
    midi_min: int,
    midi_max: int,
    midi_pitches: tuple[int, ...] | list[int] | None,
) -> np.ndarray:
    if midi_pitches is None:
        if midi_max < midi_min:
            raise ValueError("midi_max must be greater than or equal to midi_min.")
        pitches = np.arange(midi_min, midi_max + 1, dtype=int)
    else:
        pitches = np.asarray(tuple(int(pitch) for pitch in midi_pitches), dtype=int)
        if np.unique(pitches).size != pitches.size:
            raise ValueError("Explicit MIDI pitches must be unique.")
    if pitches.size == 0:
        raise ValueError("At least one pitch is required.")
    return pitches


def _normalise_sequence_pitches(
    sequence_pitches: tuple[tuple[int, ...], ...] | list[tuple[int, ...]] | None,
) -> tuple[tuple[int, ...], ...] | None:
    if sequence_pitches is None:
        return None
    sequences = tuple(tuple(int(pitch) for pitch in sequence) for sequence in sequence_pitches)
    if not sequences:
        raise ValueError("At least one sequence is required.")
    if any(not sequence for sequence in sequences):
        raise ValueError("Sequences must not be empty.")
    return sequences
