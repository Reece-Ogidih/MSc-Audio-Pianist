"""Transcription metrics for the indirect audio-to-action pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from ala_pianist.music.timed_notes import TimedNote


@dataclass(frozen=True)
class TranscriptionMetrics:
    expected_count: int
    predicted_count: int
    matched_count: int
    note_precision: float
    note_recall: float
    note_f1: float
    onset_mae_seconds: float | None
    offset_mae_seconds: float | None
    duration_mae_seconds: float | None
    false_positive_count: int
    false_negative_count: int
    false_positive_pitches: tuple[int, ...]
    false_negative_pitches: tuple[int, ...]
    per_pitch: dict[int, dict[str, float | int]]

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "expected_count": self.expected_count,
            "predicted_count": self.predicted_count,
            "matched_count": self.matched_count,
            "note_precision": self.note_precision,
            "note_recall": self.note_recall,
            "note_f1": self.note_f1,
            "onset_mae_seconds": self.onset_mae_seconds,
            "offset_mae_seconds": self.offset_mae_seconds,
            "duration_mae_seconds": self.duration_mae_seconds,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "false_positive_pitches": list(self.false_positive_pitches),
            "false_negative_pitches": list(self.false_negative_pitches),
            "per_pitch": self.per_pitch,
        }


def transcription_note_metrics(
    expected: Iterable[TimedNote],
    predicted: Iterable[TimedNote],
    *,
    onset_tolerance_seconds: float = 0.05,
    offset_tolerance_seconds: float | None = None,
) -> TranscriptionMetrics:
    """Compute note-level precision/recall/F1 and timing errors."""

    expected_notes = sorted(expected, key=lambda note: (note.onset, note.offset, note.pitch))
    predicted_notes = sorted(predicted, key=lambda note: (note.onset, note.offset, note.pitch))
    if onset_tolerance_seconds < 0.0:
        raise ValueError("onset_tolerance_seconds must be non-negative.")
    if offset_tolerance_seconds is not None and offset_tolerance_seconds < 0.0:
        raise ValueError("offset_tolerance_seconds must be non-negative.")

    used_predicted: set[int] = set()
    matches: list[tuple[TimedNote, TimedNote]] = []
    matched_expected: set[int] = set()
    for expected_index, expected_note in enumerate(expected_notes):
        candidate_index = None
        candidate_error = float("inf")
        for index, predicted_note in enumerate(predicted_notes):
            if index in used_predicted or predicted_note.pitch != expected_note.pitch:
                continue
            onset_error = abs(predicted_note.onset - expected_note.onset)
            if onset_error > onset_tolerance_seconds:
                continue
            if offset_tolerance_seconds is not None:
                offset_error = abs(predicted_note.offset - expected_note.offset)
                if offset_error > offset_tolerance_seconds:
                    continue
            if onset_error < candidate_error:
                candidate_error = onset_error
                candidate_index = index
        if candidate_index is not None:
            used_predicted.add(candidate_index)
            matched_expected.add(expected_index)
            matches.append((expected_note, predicted_notes[candidate_index]))

    false_positive_notes = [
        note for index, note in enumerate(predicted_notes) if index not in used_predicted
    ]
    false_negative_notes = [
        note for index, note in enumerate(expected_notes) if index not in matched_expected
    ]
    precision = _safe_precision(len(matches), len(predicted_notes), len(expected_notes))
    recall = _safe_recall(len(matches), len(expected_notes), len(predicted_notes))
    f1 = _f1(precision, recall)
    return TranscriptionMetrics(
        expected_count=len(expected_notes),
        predicted_count=len(predicted_notes),
        matched_count=len(matches),
        note_precision=precision,
        note_recall=recall,
        note_f1=f1,
        onset_mae_seconds=_mae(abs(obs.onset - exp.onset) for exp, obs in matches),
        offset_mae_seconds=_mae(abs(obs.offset - exp.offset) for exp, obs in matches),
        duration_mae_seconds=_mae(abs(obs.duration - exp.duration) for exp, obs in matches),
        false_positive_count=len(false_positive_notes),
        false_negative_count=len(false_negative_notes),
        false_positive_pitches=tuple(note.pitch for note in false_positive_notes),
        false_negative_pitches=tuple(note.pitch for note in false_negative_notes),
        per_pitch=_per_pitch_metrics(expected_notes, predicted_notes, matches),
    )


def _safe_precision(tp: int, predicted: int, expected: int) -> float:
    if predicted == 0 and expected == 0:
        return 1.0
    if predicted == 0:
        return 0.0
    return tp / predicted


def _safe_recall(tp: int, expected: int, predicted: int) -> float:
    if expected == 0 and predicted == 0:
        return 1.0
    if expected == 0:
        return 0.0
    return tp / expected


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _mae(values: Iterable[float]) -> float | None:
    finite_values = [float(value) for value in values if isfinite(float(value))]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _per_pitch_metrics(
    expected: list[TimedNote],
    predicted: list[TimedNote],
    matches: list[tuple[TimedNote, TimedNote]],
) -> dict[int, dict[str, float | int]]:
    pitches = sorted({note.pitch for note in expected} | {note.pitch for note in predicted})
    matched_expected_counts: dict[int, int] = {}
    for expected_note, _ in matches:
        matched_expected_counts[expected_note.pitch] = matched_expected_counts.get(expected_note.pitch, 0) + 1
    result = {}
    for pitch in pitches:
        expected_count = sum(1 for note in expected if note.pitch == pitch)
        predicted_count = sum(1 for note in predicted if note.pitch == pitch)
        tp = matched_expected_counts.get(pitch, 0)
        precision = _safe_precision(tp, predicted_count, expected_count)
        recall = _safe_recall(tp, expected_count, predicted_count)
        result[int(pitch)] = {
            "expected_count": expected_count,
            "predicted_count": predicted_count,
            "matched_count": tp,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    return result
