"""Small deterministic MIDI helpers for ALA Pianist diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from note_seq import midi_io, music_pb2


@dataclass(frozen=True)
class NoteEvent:
    """A single monophonic MIDI note event."""

    pitch: int
    start: float
    duration: float
    velocity: int = 80
    fingering: int | None = None


def write_monophonic_midi(
    note_events: Iterable[NoteEvent],
    path: str | Path,
    *,
    qpm: float = 90.0,
    title: str = "ALA Pianist monophonic clip",
) -> Path:
    """Write a short monophonic `.mid` file from deterministic note events."""

    events = list(note_events)
    if not events:
        raise ValueError("At least one note event is required.")

    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = title
    seq.sequence_metadata.artist = "ALA Pianist"
    seq.tempos.add(qpm=qpm)

    previous_end = -1.0
    total_time = 0.0
    for event in events:
        if event.duration <= 0:
            raise ValueError(f"Note duration must be positive, got {event.duration}.")
        if event.start < previous_end:
            raise ValueError("Note events must be monophonic and non-overlapping.")
        if not 0 <= event.velocity <= 127:
            raise ValueError(f"Velocity must be in [0, 127], got {event.velocity}.")
        if event.fingering is not None and not 0 <= event.fingering <= 9:
            raise ValueError(
                f"Fingering labels must be in [0, 9], got {event.fingering}."
            )

        end = event.start + event.duration
        seq.notes.add(
            pitch=int(event.pitch),
            start_time=float(event.start),
            end_time=float(end),
            velocity=int(event.velocity),
            part=0 if event.fingering is None else int(event.fingering),
        )
        previous_end = end
        total_time = max(total_time, end)

    seq.total_time = total_time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi_io.note_sequence_to_midi_file(seq, str(path))
    return path
