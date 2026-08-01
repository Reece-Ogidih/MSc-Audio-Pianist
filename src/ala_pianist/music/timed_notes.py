"""Canonical timed-note representation for indirect audio-to-action pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from note_seq import midi_io, music_pb2

from ala_pianist.music.midi_utils import NoteEvent, write_monophonic_midi


RangePolicy = Literal["error", "drop", "clip"]
DuplicatePolicy = Literal["keep", "merge", "error"]


@dataclass(frozen=True)
class TimedNote:
    """A model-independent symbolic note with explicit timing and confidence."""

    pitch: int
    onset: float
    offset: float
    confidence: float = 1.0
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= int(self.pitch) <= 127:
            raise ValueError(f"pitch must be in [0, 127], got {self.pitch}.")
        if float(self.onset) < 0.0:
            raise ValueError(f"onset must be non-negative, got {self.onset}.")
        if float(self.offset) <= float(self.onset):
            raise ValueError("offset must be greater than onset.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}.")

    @property
    def duration(self) -> float:
        return float(self.offset) - float(self.onset)

    @property
    def key_index(self) -> int:
        return int(self.pitch) - 21


@dataclass(frozen=True)
class ControllerGoalSequence:
    """Controller-compatible symbolic goal sequence with preserved note timing."""

    notes: tuple[TimedNote, ...]
    note_events: tuple[NoteEvent, ...]
    midi_min: int
    midi_max: int
    timing_preserved: bool = True
    monophonic: bool = True

    @property
    def pitches(self) -> tuple[int, ...]:
        return tuple(note.pitch for note in self.notes)

    @property
    def key_indices(self) -> tuple[int, ...]:
        return tuple(note.key_index for note in self.notes)


@dataclass(frozen=True)
class MonophonicCleanupConfig:
    """Explicit cleanup policy for monophonic transcription outputs."""

    audio_duration_seconds: float | None = None
    suppress_repeated_same_pitch: bool = True
    truncate_overlaps: bool = True
    preserve_onsets: bool = True


@dataclass(frozen=True)
class MonophonicCleanupDiagnostics:
    raw_note_count: int
    cleaned_note_count: int
    invalid_duration_count: int
    clipped_to_audio_duration_count: int
    same_pitch_duplicate_suppressed_count: int
    overlap_truncation_count: int
    raw_overlap_count: int
    cleaned_overlap_count: int
    raw_same_pitch_duplicate_count: int
    cleaned_same_pitch_duplicate_count: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def note_event_to_timed_note(
    event: NoteEvent,
    *,
    confidence: float = 1.0,
    source: str = "midi",
    metadata: dict[str, Any] | None = None,
) -> TimedNote:
    """Convert the existing MIDI helper event into the canonical timed-note form."""

    return TimedNote(
        pitch=int(event.pitch),
        onset=float(event.start),
        offset=float(event.start + event.duration),
        confidence=float(confidence),
        source=source,
        metadata=dict(metadata or {}),
    )


def timed_note_to_note_event(
    note: TimedNote,
    *,
    velocity: int = 90,
    fingering: int | None = None,
) -> NoteEvent:
    """Convert a canonical timed note to RoboPianist-compatible MIDI event timing."""

    return NoteEvent(
        pitch=int(note.pitch),
        start=float(note.onset),
        duration=float(note.duration),
        velocity=int(velocity),
        fingering=fingering,
    )


def normalize_timed_notes(
    notes: Iterable[TimedNote],
    *,
    midi_min: int = 72,
    midi_max: int = 76,
    confidence_threshold: float = 0.0,
    range_policy: RangePolicy = "error",
    duplicate_policy: DuplicatePolicy = "merge",
    allow_polyphony: bool = True,
) -> tuple[TimedNote, ...]:
    """Filter and validate timed notes without silently quantizing timing."""

    if midi_min > midi_max:
        raise ValueError("midi_min must be <= midi_max.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1].")

    accepted: list[TimedNote] = []
    for note in notes:
        if note.confidence < confidence_threshold:
            continue
        ranged_note = _apply_range_policy(note, midi_min, midi_max, range_policy)
        if ranged_note is not None:
            accepted.append(ranged_note)
    accepted.sort(key=lambda note: (note.onset, note.offset, note.pitch))
    if not allow_polyphony:
        _validate_monophonic(accepted)
    return _handle_duplicates(accepted, duplicate_policy)


def timed_notes_to_controller_sequence(
    notes: Iterable[TimedNote],
    *,
    midi_min: int = 72,
    midi_max: int = 76,
    confidence_threshold: float = 0.0,
    range_policy: RangePolicy = "error",
    duplicate_policy: DuplicatePolicy = "merge",
    allow_polyphony: bool = True,
    velocity: int = 90,
) -> ControllerGoalSequence:
    """Build the symbolic sequence consumed downstream by controller rollouts."""

    normalized = normalize_timed_notes(
        notes,
        midi_min=midi_min,
        midi_max=midi_max,
        confidence_threshold=confidence_threshold,
        range_policy=range_policy,
        duplicate_policy=duplicate_policy,
        allow_polyphony=allow_polyphony,
    )
    events = tuple(timed_note_to_note_event(note, velocity=velocity) for note in normalized)
    return ControllerGoalSequence(
        notes=normalized,
        note_events=events,
        midi_min=int(midi_min),
        midi_max=int(midi_max),
        monophonic=_is_monophonic(normalized),
    )


def write_controller_goal_midi(sequence: ControllerGoalSequence, path: str | Path) -> Path:
    """Write a controller-ready MIDI file from a preserved-timing goal sequence."""

    if not sequence.note_events:
        raise ValueError("Cannot write an empty controller goal MIDI.")
    if sequence.monophonic:
        return write_monophonic_midi(
            sequence.note_events,
            path,
            title="ALA Pianist indirect controller goal",
        )
    return write_timed_notes_midi(sequence.notes, path, title="ALA Pianist indirect controller goal")


def write_timed_notes_midi(
    notes: Iterable[TimedNote],
    path: str | Path,
    *,
    qpm: float = 90.0,
    title: str = "ALA Pianist timed-note clip",
    default_velocity: int = 90,
) -> Path:
    """Write timed notes to MIDI, preserving overlaps as explicit predictions."""

    events = tuple(notes)
    if not events:
        raise ValueError("At least one timed note is required.")
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = title
    seq.sequence_metadata.artist = "ALA Pianist"
    seq.tempos.add(qpm=float(qpm))
    total_time = 0.0
    for note in events:
        velocity = int(note.metadata.get("velocity", default_velocity))
        seq.notes.add(
            pitch=int(note.pitch),
            start_time=float(note.onset),
            end_time=float(note.offset),
            velocity=max(0, min(127, velocity)),
        )
        total_time = max(total_time, float(note.offset))
    seq.total_time = total_time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi_io.note_sequence_to_midi_file(seq, str(path))
    return path


def clean_monophonic_timed_notes(
    notes: Iterable[TimedNote],
    *,
    config: MonophonicCleanupConfig | None = None,
) -> tuple[tuple[TimedNote, ...], MonophonicCleanupDiagnostics]:
    """Clean monophonic predictions without using oracle notes or controller results."""

    config = config or MonophonicCleanupConfig()
    raw_notes = sorted(tuple(notes), key=lambda note: (note.onset, note.offset, note.pitch))
    invalid_duration_count = 0
    clipped_to_audio_duration_count = 0
    valid_notes: list[TimedNote] = []
    for note in raw_notes:
        if note.duration <= 0.0:
            invalid_duration_count += 1
            continue
        cleaned_note = note
        if config.audio_duration_seconds is not None:
            audio_duration = float(config.audio_duration_seconds)
            if cleaned_note.onset >= audio_duration:
                invalid_duration_count += 1
                continue
            if cleaned_note.offset > audio_duration:
                clipped_to_audio_duration_count += 1
                cleaned_note = _replace_timed_note(
                    cleaned_note,
                    offset=audio_duration,
                    metadata_update={"cleanup_clipped_to_audio_duration": True},
                )
        if cleaned_note.duration <= 0.0:
            invalid_duration_count += 1
            continue
        valid_notes.append(cleaned_note)

    suppressed = 0
    if config.suppress_repeated_same_pitch:
        deduped: list[TimedNote] = []
        seen_pitches: set[int] = set()
        for note in valid_notes:
            if note.pitch in seen_pitches:
                suppressed += 1
                if deduped and deduped[-1].pitch == note.pitch:
                    previous = deduped[-1]
                    deduped[-1] = _replace_timed_note(
                        previous,
                        confidence=max(previous.confidence, note.confidence),
                        metadata_update={
                            "cleanup_suppressed_following_duplicate": True,
                            "cleanup_suppressed_duplicate_onset": note.onset,
                            "cleanup_suppressed_duplicate_offset": note.offset,
                            "cleanup_suppressed_duplicate_confidence": note.confidence,
                        },
                    )
                continue
            seen_pitches.add(note.pitch)
            deduped.append(note)
        valid_notes = deduped

    overlap_truncations = 0
    if config.truncate_overlaps:
        truncated: list[TimedNote] = []
        for index, note in enumerate(valid_notes):
            next_note = valid_notes[index + 1] if index + 1 < len(valid_notes) else None
            if next_note is not None and note.offset > next_note.onset:
                overlap_truncations += 1
                note = _replace_timed_note(
                    note,
                    offset=max(note.onset, next_note.onset),
                    metadata_update={
                        "cleanup_truncated_at_next_onset": True,
                        "cleanup_next_onset": next_note.onset,
                    },
                )
            if note.duration > 0.0:
                truncated.append(note)
            else:
                invalid_duration_count += 1
        valid_notes = truncated

    cleaned = tuple(sorted(valid_notes, key=lambda note: (note.onset, note.offset, note.pitch)))
    diagnostics = MonophonicCleanupDiagnostics(
        raw_note_count=len(raw_notes),
        cleaned_note_count=len(cleaned),
        invalid_duration_count=invalid_duration_count,
        clipped_to_audio_duration_count=clipped_to_audio_duration_count,
        same_pitch_duplicate_suppressed_count=suppressed,
        overlap_truncation_count=overlap_truncations,
        raw_overlap_count=count_overlapping_notes(raw_notes),
        cleaned_overlap_count=count_overlapping_notes(cleaned),
        raw_same_pitch_duplicate_count=count_repeated_same_pitch_notes(raw_notes),
        cleaned_same_pitch_duplicate_count=count_repeated_same_pitch_notes(cleaned),
    )
    return cleaned, diagnostics


def count_overlapping_notes(notes: Iterable[TimedNote]) -> int:
    """Count notes whose offset extends beyond the next note onset."""

    ordered = sorted(tuple(notes), key=lambda note: (note.onset, note.offset, note.pitch))
    count = 0
    for index, note in enumerate(ordered[:-1]):
        if note.offset > ordered[index + 1].onset:
            count += 1
    return count


def count_repeated_same_pitch_notes(notes: Iterable[TimedNote]) -> int:
    """Count later predictions for a pitch already present in the sequence."""

    seen: set[int] = set()
    count = 0
    for note in sorted(tuple(notes), key=lambda item: (item.onset, item.offset, item.pitch)):
        if note.pitch in seen:
            count += 1
        seen.add(note.pitch)
    return count


def timing_quantization_report(
    sequence: ControllerGoalSequence,
    *,
    control_timestep_seconds: float,
) -> dict[str, float | int]:
    """Report explicit control-step timing discretisation error."""

    if control_timestep_seconds <= 0.0:
        raise ValueError("control_timestep_seconds must be positive.")
    errors = []
    for note in sequence.notes:
        for value in (note.onset, note.offset):
            quantized = round(value / control_timestep_seconds) * control_timestep_seconds
            errors.append(abs(quantized - value))
    return {
        "control_timestep_seconds": float(control_timestep_seconds),
        "event_count": len(errors),
        "max_abs_error_seconds": max(errors, default=0.0),
        "mean_abs_error_seconds": sum(errors) / len(errors) if errors else 0.0,
    }


def _apply_range_policy(
    note: TimedNote,
    midi_min: int,
    midi_max: int,
    policy: RangePolicy,
) -> TimedNote | None:
    if midi_min <= note.pitch <= midi_max:
        return note
    if policy == "drop":
        return None
    if policy == "clip":
        clipped_pitch = min(max(note.pitch, midi_min), midi_max)
        metadata = dict(note.metadata)
        metadata["original_pitch"] = int(note.pitch)
        metadata["range_policy"] = "clip"
        return TimedNote(
            pitch=clipped_pitch,
            onset=note.onset,
            offset=note.offset,
            confidence=note.confidence,
            source=note.source,
            metadata=metadata,
        )
    if policy == "error":
        raise ValueError(
            f"pitch {note.pitch} is outside supported range [{midi_min}, {midi_max}]."
        )
    raise ValueError(f"Unknown range_policy: {policy}.")


def _replace_timed_note(
    note: TimedNote,
    *,
    onset: float | None = None,
    offset: float | None = None,
    confidence: float | None = None,
    metadata_update: dict[str, Any] | None = None,
) -> TimedNote:
    metadata = dict(note.metadata)
    metadata.update(metadata_update or {})
    return TimedNote(
        pitch=note.pitch,
        onset=note.onset if onset is None else float(onset),
        offset=note.offset if offset is None else float(offset),
        confidence=note.confidence if confidence is None else float(confidence),
        source=note.source,
        metadata=metadata,
    )


def _validate_monophonic(notes: list[TimedNote]) -> None:
    previous: TimedNote | None = None
    for note in notes:
        if previous is not None and note.onset < previous.offset:
            same_duplicate = note.pitch == previous.pitch and note.onset == previous.onset
            if not same_duplicate:
                raise ValueError("Timed notes must be monophonic and non-overlapping.")
        previous = note


def _is_monophonic(notes: tuple[TimedNote, ...]) -> bool:
    previous: TimedNote | None = None
    for note in sorted(notes, key=lambda item: (item.onset, item.offset, item.pitch)):
        if previous is not None and note.onset < previous.offset:
            return False
        previous = note
    return True


def _handle_duplicates(notes: list[TimedNote], policy: DuplicatePolicy) -> tuple[TimedNote, ...]:
    if policy == "keep":
        return tuple(notes)
    if policy not in {"merge", "error"}:
        raise ValueError(f"Unknown duplicate_policy: {policy}.")

    result: list[TimedNote] = []
    for note in notes:
        duplicate = (
            result
            and result[-1].pitch == note.pitch
            and result[-1].onset == note.onset
            and result[-1].offset == note.offset
        )
        if duplicate and policy == "error":
            raise ValueError("Duplicate timed note found.")
        if duplicate:
            previous = result[-1]
            metadata = dict(previous.metadata)
            metadata["duplicate_count"] = int(metadata.get("duplicate_count", 1)) + 1
            result[-1] = TimedNote(
                pitch=previous.pitch,
                onset=previous.onset,
                offset=previous.offset,
                confidence=max(previous.confidence, note.confidence),
                source=previous.source,
                metadata=metadata,
            )
        else:
            result.append(note)
    return tuple(result)
