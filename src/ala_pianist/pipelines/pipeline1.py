"""Rough Pipeline 1: audio -> MIDI -> action -> RoboPianist rollout -> metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ala_pianist.audio import synthesize_monophonic_wav, transcribe_monophonic_wav, transcription_accuracy
from ala_pianist.controllers import ActionLibraryEntry
from ala_pianist.controllers.action_library import KEYSET_MIDI, build_action_library, load_action_library
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


@dataclass(frozen=True)
class NoteRolloutMetric:
    midi_pitch: int
    key_index: int
    max_target_key_state: float
    max_unintended_key_state: float
    pressed_keys: tuple[int, ...]
    outcome: str
    strict_outcome: str
    trajectory_quality: str
    nearby_key_states: dict[int, float]
    clean_press: bool
    dirty_press: bool
    missed: bool
    closest_finger_to_target: str
    finger_target_distance: float | None
    target_contact_finger: str
    contact_pairs: tuple[str, ...]
    fingering_note: str


@dataclass(frozen=True)
class Pipeline1Result:
    expected_pitches: tuple[int, ...]
    transcribed_pitches: tuple[int, ...]
    transcription: dict
    note_metrics: tuple[NoteRolloutMetric, ...]
    target_recall: float
    wrong_key_rate: float
    max_unintended_key_state: float
    clean_hit_count: int
    dirty_hit_count: int
    miss_count: int


def default_pipeline1_events() -> list[NoteEvent]:
    return [
        NoteEvent(69, 0.00, 0.28, 90),
        NoteEvent(73, 0.40, 0.28, 90),
        NoteEvent(75, 0.80, 0.28, 90),
        NoteEvent(71, 1.20, 0.28, 90),
    ]


def pipeline1_events_from_pitches(pitches: list[int] | tuple[int, ...]) -> list[NoteEvent]:
    return [
        NoteEvent(int(pitch), 0.40 * index, 0.28, 90)
        for index, pitch in enumerate(pitches)
    ]


def run_pipeline1(
    *,
    audio_path: str | Path,
    library_path: str | Path,
    rollout_midi_path: str | Path,
    summary_path: str | Path | None = None,
    note_events: list[NoteEvent] | None = None,
    build_library: bool = True,
    controller: Any | None = None,
) -> Pipeline1Result:
    """Run the first rough Pipeline 1 diagnostic end to end."""

    expected_events = note_events or default_pipeline1_events()
    synthesize_monophonic_wav(expected_events, audio_path)
    transcription = transcribe_monophonic_wav(audio_path)
    accuracy = transcription_accuracy(expected_events, transcription.note_events)

    if build_library or not Path(library_path).exists():
        library = build_action_library(library_path, midi_pitches=KEYSET_MIDI)
    else:
        library = load_action_library(library_path)

    transcribed_events = [event for event in transcription.note_events if event.pitch in library]
    if not transcribed_events:
        raise RuntimeError("No transcribed events were available in the action library key set.")
    write_monophonic_midi(transcribed_events, rollout_midi_path, title="pipeline1 transcribed rollout")
    env = ALAOneHandEnv(rollout_midi_path)
    env.reset()

    metrics = []
    for event in transcribed_events:
        entry = library[event.pitch]
        metrics.append(_rollout_note(env, entry, horizon_steps=24, controller=controller))

    hit_count = sum(metric.max_target_key_state >= 0.25 or metric.key_index in metric.pressed_keys for metric in metrics)
    wrong_count = sum(bool([key for key in metric.pressed_keys if key != metric.key_index]) for metric in metrics)
    clean_count = sum(metric.clean_press for metric in metrics)
    dirty_count = sum(metric.dirty_press for metric in metrics)
    miss_count = sum(metric.missed for metric in metrics)
    result = Pipeline1Result(
        expected_pitches=tuple(event.pitch for event in expected_events),
        transcribed_pitches=tuple(event.pitch for event in transcription.note_events),
        transcription=accuracy,
        note_metrics=tuple(metrics),
        target_recall=hit_count / max(1, len(metrics)),
        wrong_key_rate=wrong_count / max(1, len(metrics)),
        max_unintended_key_state=max((metric.max_unintended_key_state for metric in metrics), default=0.0),
        clean_hit_count=clean_count,
        dirty_hit_count=dirty_count,
        miss_count=miss_count,
    )
    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def _rollout_note(
    env: ALAOneHandEnv,
    entry: ActionLibraryEntry,
    *,
    horizon_steps: int,
    controller: Any | None = None,
) -> NoteRolloutMetric:
    library_action = np.asarray(entry.action, dtype=env.action_spec().dtype)
    max_target = 0.0
    max_unintended = 0.0
    nearby_key_states = {key: 0.0 for key in _nearby_keys(entry.key_index)}
    pressed = set()
    contact_pairs = set()
    closest_finger = "unknown"
    closest_distance = None
    target_contact_finger = "unknown"
    for step in range(horizon_steps):
        if controller is None:
            ramp = min(1.0, (step + 1) / 6.0)
            action = library_action * ramp
        else:
            action = controller.action(
                env,
                target_midi=entry.midi_pitch,
                fallback_action=library_action,
                step_count=step,
            )
        timestep = env.step(action)
        max_target = max(max_target, env.target_key_state(entry.key_index) or 0.0)
        max_unintended = max(max_unintended, env.max_unintended_key_state(entry.key_index))
        for key in nearby_key_states:
            nearby_key_states[key] = max(nearby_key_states[key], env.target_key_state(key) or 0.0)
        pressed.update(env.current_pressed_keys())
        closest = env.nearest_fingertip_to_key(entry.key_index)
        if closest is not None:
            closest_finger = str(closest["fingertip"])
            distance = float(closest["distance"])
            closest_distance = distance if closest_distance is None else min(closest_distance, distance)
        for pair in env.key_contact_pairs(entry.key_index):
            formatted = f"{pair[0]} <-> {pair[1]}"
            contact_pairs.add(formatted)
            target_contact_finger = _finger_from_contact_pair(pair)
        if timestep.last():
            break
    outcome = _outcome(entry.key_index, pressed, max_target, max_unintended)
    clean = pressed == {entry.key_index}
    dirty = entry.key_index in pressed and not clean
    missed = not clean and not dirty and not (max_target >= 0.25 and max_unintended <= max_target + 0.02)
    strict_outcome = _strict_outcome(entry.key_index, pressed, max_target, max_unintended)
    return NoteRolloutMetric(
        midi_pitch=entry.midi_pitch,
        key_index=entry.key_index,
        max_target_key_state=float(max_target),
        max_unintended_key_state=float(max_unintended),
        pressed_keys=tuple(sorted(pressed)),
        outcome=outcome,
        strict_outcome=strict_outcome,
        trajectory_quality=_trajectory_quality(strict_outcome, max_unintended),
        nearby_key_states={int(key): float(value) for key, value in nearby_key_states.items()},
        clean_press=clean,
        dirty_press=dirty,
        missed=missed,
        closest_finger_to_target=closest_finger,
        finger_target_distance=closest_distance,
        target_contact_finger=target_contact_finger,
        contact_pairs=tuple(sorted(contact_pairs)),
        fingering_note=_fingering_note(target_contact_finger, closest_finger, closest_distance),
    )


def _outcome(target_key: int, pressed: set[int], max_target: float, max_unintended: float) -> str:
    if pressed == {target_key}:
        return "clean"
    if target_key in pressed:
        return "dirty"
    if max_target >= 0.25 and max_unintended <= max_target + 0.02:
        return "near_clean"
    if max_target > 0.02:
        return "partial"
    return "missed"


def _strict_outcome(target_key: int, pressed: set[int], max_target: float, max_unintended: float) -> str:
    if pressed == {target_key} and max_unintended < 0.25:
        return "clean_low_unintended"
    if pressed == {target_key}:
        return "clean_high_unintended"
    if target_key in pressed:
        return "dirty_pressed_wrong_key"
    if max_target >= 0.25 and max_unintended < 0.25:
        return "near_clean_partial"
    return "missed"


def _trajectory_quality(strict_outcome: str, max_unintended: float) -> str:
    if strict_outcome == "clean_low_unintended":
        return "gold_demo_candidate"
    if strict_outcome == "clean_high_unintended" and max_unintended < 0.80:
        return "weak_demo_candidate"
    return "not_demo_candidate"


def _nearby_keys(target_key: int) -> tuple[int, ...]:
    return tuple(key for key in range(max(0, target_key - 2), min(87, target_key + 2) + 1))


def _finger_from_contact_pair(pair: tuple[str, str]) -> str:
    text = " ".join(pair)
    for token, name in (
        ("th", "thumb"),
        ("ff", "index"),
        ("mf", "middle"),
        ("rf", "ring"),
        ("lf", "little"),
    ):
        if token in text:
            return name
    return "unknown"


def _fingering_note(contact_finger: str, closest_finger: str, distance: float | None) -> str:
    if contact_finger != "unknown":
        return f"target contact observed with {contact_finger}; diagnostic only"
    if closest_finger != "unknown":
        return f"no reliable target contact finger; closest site={closest_finger} distance={distance}"
    return "fingering unknown; contact data unavailable or inconclusive"
