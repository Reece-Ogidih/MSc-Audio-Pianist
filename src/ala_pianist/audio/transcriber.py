"""Model-independent audio-to-MIDI transcription interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from note_seq import midi_io

from ala_pianist.audio.transcription import transcribe_monophonic_wav
from ala_pianist.music.timed_notes import TimedNote, note_event_to_timed_note


@dataclass(frozen=True)
class TranscriptionOutput:
    """Canonical output shared by oracle and learned transcribers."""

    notes: tuple[TimedNote, ...]
    transcriber_name: str
    source_audio_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AudioToMidiTranscriber(Protocol):
    """Swappable interface for audio-to-MIDI systems."""

    name: str

    def transcribe(self, audio_path: str | Path) -> TranscriptionOutput:
        """Return canonical timed-note predictions for an audio file."""


@dataclass(frozen=True)
class GeneratedWavPeakTranscriber:
    """Adapter for the existing deterministic generated-WAV pitch transcriber."""

    name: str = "generated_wav_peak"

    def transcribe(self, audio_path: str | Path) -> TranscriptionOutput:
        result = transcribe_monophonic_wav(audio_path)
        notes = tuple(
            note_event_to_timed_note(
                event,
                confidence=1.0,
                source=self.name,
                metadata={"adapter": "transcribe_monophonic_wav"},
            )
            for event in result.note_events
        )
        return TranscriptionOutput(
            notes=notes,
            transcriber_name=self.name,
            source_audio_path=Path(audio_path),
            metadata={"frame_count": len(result.frame_pitches)},
        )


@dataclass(frozen=True)
class OracleMidiTranscriber:
    """Ground-truth MIDI adapter using the same downstream transcriber contract."""

    midi_path: Path
    name: str = "oracle_midi"

    def __init__(self, midi_path: str | Path, name: str = "oracle_midi") -> None:
        object.__setattr__(self, "midi_path", Path(midi_path))
        object.__setattr__(self, "name", name)

    def transcribe(self, audio_path: str | Path | None = None) -> TranscriptionOutput:
        sequence = midi_io.midi_file_to_note_sequence(str(self.midi_path))
        notes = []
        for index, note in enumerate(sequence.notes):
            notes.append(
                TimedNote(
                    pitch=int(note.pitch),
                    onset=float(note.start_time),
                    offset=float(note.end_time),
                    confidence=1.0,
                    source=self.name,
                    metadata={
                        "midi_path": str(self.midi_path),
                        "note_index": index,
                        "velocity": int(note.velocity),
                    },
                )
            )
        notes.sort(key=lambda note: (note.onset, note.offset, note.pitch))
        return TranscriptionOutput(
            notes=tuple(notes),
            transcriber_name=self.name,
            source_audio_path=None if audio_path is None else Path(audio_path),
            metadata={"midi_path": str(self.midi_path), "note_count": len(notes)},
        )


@dataclass(frozen=True)
class BasicPitchTranscriber:
    """Spotify Basic Pitch adapter returning canonical timed notes."""

    model_path: Path | None = None
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    minimum_note_length_ms: float = 80.0
    minimum_frequency: float | None = None
    maximum_frequency: float | None = None
    melodia_trick: bool = True
    name: str = "basic_pitch"

    def transcribe(self, audio_path: str | Path) -> TranscriptionOutput:
        try:
            from basic_pitch import inference as basic_pitch_inference
        except Exception as exc:  # pragma: no cover - exercised when dependency absent.
            raise RuntimeError(
                "Basic Pitch is not available. Install basic-pitch with a supported inference runtime."
            ) from exc

        model_path = self.model_path
        if model_path is None:
            model_path = Path(basic_pitch_inference.ICASSP_2022_MODEL_PATH).with_suffix(".onnx")
        model_output, _midi_data, raw_notes = basic_pitch_inference.predict(
            audio_path,
            model_or_model_path=model_path,
            onset_threshold=self.onset_threshold,
            frame_threshold=self.frame_threshold,
            minimum_note_length=self.minimum_note_length_ms,
            minimum_frequency=self.minimum_frequency,
            maximum_frequency=self.maximum_frequency,
            multiple_pitch_bends=False,
            melodia_trick=self.melodia_trick,
        )
        notes, invalid_notes = basic_pitch_notes_to_timed_notes(raw_notes, source=self.name)
        return TranscriptionOutput(
            notes=tuple(notes),
            transcriber_name=self.name,
            source_audio_path=Path(audio_path),
            metadata={
                "onset_threshold": self.onset_threshold,
                "frame_threshold": self.frame_threshold,
                "minimum_note_length_ms": self.minimum_note_length_ms,
                "minimum_frequency": self.minimum_frequency,
                "maximum_frequency": self.maximum_frequency,
                "melodia_trick": self.melodia_trick,
                "model_path": str(model_path),
                "raw_note_count": len(raw_notes),
                "invalid_note_count": invalid_notes,
                "model_output_keys": sorted(str(key) for key in model_output.keys()),
            },
        )


def basic_pitch_notes_to_timed_notes(
    raw_notes,
    *,
    source: str = "basic_pitch",
) -> tuple[tuple[TimedNote, ...], int]:
    """Convert Basic Pitch raw note tuples to canonical timed notes."""

    notes = []
    invalid_notes = 0
    for index, raw_note in enumerate(raw_notes):
        onset, offset, pitch, amplitude, pitch_bends = raw_note
        if not np.isfinite(onset) or not np.isfinite(offset) or float(offset) <= float(onset):
            invalid_notes += 1
            continue
        notes.append(
            TimedNote(
                pitch=int(pitch),
                onset=float(onset),
                offset=float(offset),
                confidence=float(max(0.0, min(1.0, amplitude))),
                source=source,
                metadata={
                    "raw_note_index": index,
                    "amplitude": float(amplitude),
                    "pitch_bends": None
                    if pitch_bends is None
                    else [int(value) for value in pitch_bends],
                },
            )
        )
    notes.sort(key=lambda note: (note.onset, note.offset, note.pitch))
    return tuple(notes), invalid_notes
