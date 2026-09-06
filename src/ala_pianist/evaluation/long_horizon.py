"""Long-horizon compositional evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ala_pianist.evaluation.metrics import binary_key_vector, pressed_key_metrics, timestep_key_metrics
from ala_pianist.music.sequence_generation import SequenceTimingConfig, generate_sequence_events
from ala_pianist.music.timed_notes import TimedNote, note_event_to_timed_note


@dataclass(frozen=True)
class LongHorizonSequence:
    """One deterministic long-horizon benchmark item."""

    name: str
    pitches: tuple[int, ...]
    archetype: str
    length: int

    @property
    def key_indices(self) -> tuple[int, ...]:
        return tuple(pitch - 21 for pitch in self.pitches)


@dataclass(frozen=True)
class LongHorizonBenchmark:
    """Manifest plus timing for long-horizon evaluation."""

    benchmark_name: str
    seed: int
    midi_pitches: tuple[int, ...]
    timing: SequenceTimingConfig
    sequences: tuple[LongHorizonSequence, ...]


def load_long_horizon_benchmark(
    path: str | Path,
    *,
    allow_trained_short: bool = False,
) -> LongHorizonBenchmark:
    """Load and validate the tracked long-horizon manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    timing = SequenceTimingConfig(
        note_duration=float(payload["note_duration"]),
        note_gap=float(payload["note_gap"]),
        velocity=int(payload["velocity"]),
    )
    allowed = set(int(pitch) for pitch in payload["midi_pitches"])
    trained_short = {(pitch,) for pitch in allowed} | {
        (left, right) for left in allowed for right in allowed if left != right
    }
    sequences = []
    for item in payload["sequences"]:
        pitches = tuple(int(pitch) for pitch in item["pitches"])
        if len(pitches) != int(item["length"]):
            raise ValueError(f"{item['name']} length metadata does not match pitches.")
        if not allow_trained_short and pitches in trained_short:
            raise ValueError(f"{item['name']} is an original anchor/two-note transition.")
        if not set(pitches).issubset(allowed):
            raise ValueError(f"{item['name']} contains pitches outside {sorted(allowed)}.")
        sequences.append(
            LongHorizonSequence(
                name=str(item["name"]),
                pitches=pitches,
                archetype=str(item["archetype"]),
                length=len(pitches),
            )
        )
    if len({sequence.name for sequence in sequences}) != len(sequences):
        raise ValueError("Long-horizon sequence names must be unique.")
    return LongHorizonBenchmark(
        benchmark_name=str(payload["benchmark_name"]),
        seed=int(payload["seed"]),
        midi_pitches=tuple(sorted(allowed)),
        timing=timing,
        sequences=tuple(sequences),
    )


def sequence_notes(sequence: LongHorizonSequence, timing: SequenceTimingConfig) -> tuple[TimedNote, ...]:
    """Return canonical timed notes for one long-horizon sequence."""

    events = generate_sequence_events(
        sequence.pitches,
        midi_min=min(sequence.pitches),
        midi_max=max(sequence.pitches),
        timing=timing,
    )
    return tuple(note_event_to_timed_note(event, source="long_horizon_manifest") for event in events)


def horizon_steps_for_notes(
    notes: Iterable[TimedNote],
    *,
    control_timestep_seconds: float = 0.05,
    tail_seconds: float = 0.80,
) -> int:
    """Scale RoboPianist evaluation horizon to the note sequence duration."""

    notes = tuple(notes)
    if not notes:
        return 1
    duration = max(note.offset for note in notes) + float(tail_seconds)
    return max(1, int(ceil(duration / float(control_timestep_seconds))))


def long_horizon_metrics_from_trace(
    *,
    notes: Iterable[TimedNote],
    step_times: Iterable[float],
    target_vectors: Iterable[np.ndarray],
    pressed_vectors: Iterable[np.ndarray],
    key_states: Iterable[np.ndarray],
    press_threshold: float = 0.5,
    soft_threshold: float = 0.2,
) -> dict[str, Any]:
    """Compute temporal/order metrics from a rollout trace."""

    events = tuple(notes)
    target_vectors = tuple(np.asarray(vector, dtype=bool) for vector in target_vectors)
    pressed_vectors = tuple(np.asarray(vector, dtype=bool) for vector in pressed_vectors)
    times = np.asarray(tuple(step_times), dtype=np.float64)
    states = np.asarray(tuple(key_states), dtype=np.float32)
    if states.ndim != 2 or states.shape[1] < 88:
        raise ValueError("key_states must be shaped (steps, 88).")
    if len(times) != states.shape[0]:
        raise ValueError("step_times and key_states must have the same length.")
    timestep = timestep_key_metrics(target_vectors, pressed_vectors)
    pressed_seen = {
        int(key)
        for vector in pressed_vectors
        for key in np.flatnonzero(np.asarray(vector, dtype=bool))
    }
    target_key_set = {note.key_index for note in events}
    pressed = pressed_key_metrics(target_key_set, pressed_seen)

    event_rows = []
    for index, note in enumerate(events):
        window = (times >= note.onset) & (times < note.offset)
        grace = (times >= note.onset) & (times < note.offset + 0.15)
        if not np.any(window):
            window = grace
        key_state = states[:, note.key_index]
        hit_mask = grace & (key_state >= press_threshold)
        hit = bool(np.any(hit_mask))
        first_hit_time = float(times[np.flatnonzero(hit_mask)[0]]) if hit else None
        active_values = key_state[window] if np.any(window) else np.asarray([], dtype=np.float32)
        event_rows.append(
            {
                "event_index": index,
                "pitch": int(note.pitch),
                "key_index": int(note.key_index),
                "onset": float(note.onset),
                "offset": float(note.offset),
                "hit": hit,
                "max_key_state": float(np.max(active_values)) if active_values.size else 0.0,
                "first_hit_time": first_hit_time,
                "onset_error": None if first_hit_time is None else float(first_hit_time - note.onset),
            }
        )

    hits = [bool(row["hit"]) for row in event_rows]
    first_failure = next((idx for idx, hit in enumerate(hits) if not hit), None)
    prefix_len = len(hits) if first_failure is None else int(first_failure)
    pair_hits = [bool(hits[index] and hits[index + 1]) for index in range(max(0, len(hits) - 1))]
    onset_errors = [abs(float(row["onset_error"])) for row in event_rows if row["onset_error"] is not None]
    offset_errors = _offset_errors(events, times, states, press_threshold=press_threshold)
    wrong_press_count, wrong_press_steps = _wrong_press_counts(events, times, pressed_vectors)
    above_soft, above_press = _unintended_threshold_steps(events, times, states, soft_threshold, press_threshold)
    first_third, middle_third, final_third = _third_hit_rates(hits)
    return {
        "pressed_key_precision": pressed.precision,
        "pressed_key_recall": pressed.recall,
        "pressed_key_f1": pressed.f1,
        "timestep_precision": timestep.precision,
        "timestep_recall": timestep.recall,
        "timestep_f1": timestep.f1,
        "target_event_count": len(events),
        "target_event_hit_count": int(sum(hits)),
        "target_event_hit_rate": float(sum(hits) / max(1, len(hits))),
        "ordered_event_accuracy": float(prefix_len / max(1, len(hits))),
        "transition_event_pair_accuracy": float(sum(pair_hits) / max(1, len(pair_hits))) if pair_hits else 1.0,
        "onset_timing_mae": None if not onset_errors else float(np.mean(onset_errors)),
        "offset_timing_mae": None if not offset_errors else float(np.mean(offset_errors)),
        "completed_target_event_fraction": float(sum(hits) / max(1, len(hits))),
        "strict_whole_sequence_success": bool(len(hits) > 0 and all(hits) and wrong_press_count == 0),
        "first_target_event_failure_index": -1 if first_failure is None else int(first_failure),
        "correctly_executed_prefix_length": prefix_len,
        "first_third_event_hit_rate": first_third,
        "middle_third_event_hit_rate": middle_third,
        "final_third_event_hit_rate": final_third,
        "unintended_press_count": int(wrong_press_count),
        "unintended_press_step_count": int(wrong_press_steps),
        "unintended_presses_per_target_event": float(wrong_press_count / max(1, len(events))),
        "unintended_timesteps_above_soft_threshold": int(above_soft),
        "unintended_timesteps_above_press_threshold": int(above_press),
        "event_rows": event_rows,
    }


def sequence_counts_by_length_and_archetype(sequences: Iterable[LongHorizonSequence]) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[tuple[int, str], int] = {}
    for sequence in sequences:
        grouped[(int(sequence.length), sequence.archetype)] = grouped.get((int(sequence.length), sequence.archetype), 0) + 1
    for (length, archetype), count in sorted(grouped.items()):
        rows.append({"sequence_length": length, "archetype": archetype, "sequence_count": count})
    return rows


def _active_keys_at(notes: tuple[TimedNote, ...], time_seconds: float) -> set[int]:
    return {note.key_index for note in notes if note.onset <= time_seconds < note.offset}


def _wrong_press_counts(
    notes: tuple[TimedNote, ...],
    times: np.ndarray,
    pressed_vectors: Iterable[np.ndarray],
) -> tuple[int, int]:
    previous_wrong: set[int] = set()
    crossings = 0
    wrong_steps = 0
    for time_seconds, vector in zip(times, pressed_vectors):
        active = _active_keys_at(notes, float(time_seconds))
        pressed = {int(key) for key in np.flatnonzero(np.asarray(vector, dtype=bool))}
        wrong = pressed - active
        wrong_steps += int(bool(wrong))
        crossings += len(wrong - previous_wrong)
        previous_wrong = wrong
    return crossings, wrong_steps


def _unintended_threshold_steps(
    notes: tuple[TimedNote, ...],
    times: np.ndarray,
    states: np.ndarray,
    soft_threshold: float,
    press_threshold: float,
) -> tuple[int, int]:
    above_soft = 0
    above_press = 0
    for row_index, time_seconds in enumerate(times):
        active = _active_keys_at(notes, float(time_seconds))
        unintended = np.asarray(states[row_index], dtype=np.float32).copy()
        for key in active:
            if 0 <= key < unintended.size:
                unintended[key] = 0.0
        above_soft += int(np.max(unintended) >= soft_threshold)
        above_press += int(np.max(unintended) >= press_threshold)
    return above_soft, above_press


def _offset_errors(
    notes: tuple[TimedNote, ...],
    times: np.ndarray,
    states: np.ndarray,
    *,
    press_threshold: float,
) -> list[float]:
    errors = []
    for note in notes:
        key_state = states[:, note.key_index]
        active = (times >= note.onset) & (key_state >= press_threshold)
        if not np.any(active):
            continue
        last_time = float(times[np.flatnonzero(active)[-1]])
        errors.append(abs(last_time - note.offset))
    return errors


def _third_hit_rates(hits: list[bool]) -> tuple[float, float, float]:
    if not hits:
        return (1.0, 1.0, 1.0)
    splits = np.array_split(np.asarray(hits, dtype=np.float32), 3)
    return tuple(float(np.mean(part)) if part.size else 1.0 for part in splits)
