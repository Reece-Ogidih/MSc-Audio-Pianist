"""Rough Pipeline 1: audio -> MIDI -> action -> RoboPianist rollout -> metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

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


@dataclass(frozen=True)
class Pipeline1Result:
    expected_pitches: tuple[int, ...]
    transcribed_pitches: tuple[int, ...]
    transcription: dict
    note_metrics: tuple[NoteRolloutMetric, ...]
    target_recall: float
    wrong_key_rate: float


def default_pipeline1_events() -> list[NoteEvent]:
    return [
        NoteEvent(69, 0.00, 0.28, 90),
        NoteEvent(73, 0.40, 0.28, 90),
        NoteEvent(75, 0.80, 0.28, 90),
        NoteEvent(71, 1.20, 0.28, 90),
    ]


def run_pipeline1(
    *,
    audio_path: str | Path,
    library_path: str | Path,
    rollout_midi_path: str | Path,
    summary_path: str | Path | None = None,
    note_events: list[NoteEvent] | None = None,
    build_library: bool = True,
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
        metrics.append(_rollout_note(env, entry, horizon_steps=24))

    hit_count = sum(metric.max_target_key_state >= 0.25 or metric.key_index in metric.pressed_keys for metric in metrics)
    wrong_count = sum(bool([key for key in metric.pressed_keys if key != metric.key_index]) for metric in metrics)
    result = Pipeline1Result(
        expected_pitches=tuple(event.pitch for event in expected_events),
        transcribed_pitches=tuple(event.pitch for event in transcription.note_events),
        transcription=accuracy,
        note_metrics=tuple(metrics),
        target_recall=hit_count / max(1, len(metrics)),
        wrong_key_rate=wrong_count / max(1, len(metrics)),
    )
    if summary_path is not None:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def _rollout_note(env: ALAOneHandEnv, entry: ActionLibraryEntry, *, horizon_steps: int) -> NoteRolloutMetric:
    action = np.asarray(entry.action, dtype=env.action_spec().dtype)
    max_target = 0.0
    max_unintended = 0.0
    pressed = set()
    for step in range(horizon_steps):
        ramp = min(1.0, (step + 1) / 6.0)
        timestep = env.step(action * ramp)
        max_target = max(max_target, env.target_key_state(entry.key_index) or 0.0)
        max_unintended = max(max_unintended, env.max_unintended_key_state(entry.key_index))
        pressed.update(env.current_pressed_keys())
        if timestep.last():
            break
    outcome = _outcome(entry.key_index, pressed, max_target, max_unintended)
    return NoteRolloutMetric(
        midi_pitch=entry.midi_pitch,
        key_index=entry.key_index,
        max_target_key_state=float(max_target),
        max_unintended_key_state=float(max_unintended),
        pressed_keys=tuple(sorted(pressed)),
        outcome=outcome,
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
