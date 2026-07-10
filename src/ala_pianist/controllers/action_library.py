"""Small MIDI-key to 22D action library for rough Pipeline 1 rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.learning.random_search import evaluate_action_pattern, generate_random_candidates
from ala_pianist.music import NoteEvent, write_monophonic_midi


KEYSET_MIDI = (69, 71, 73, 74, 75)


@dataclass(frozen=True)
class ActionLibraryEntry:
    midi_pitch: int
    key_index: int
    action: tuple[float, ...]
    max_target_key_state: float
    max_unintended_key_state: float
    pressed_keys: tuple[int, ...]
    outcome: str
    debug_return: float


def build_action_library(
    path: str | Path,
    *,
    midi_pitches: tuple[int, ...] = KEYSET_MIDI,
    candidate_count: int = 8,
    horizon_steps: int = 24,
    seed: int = 101,
) -> dict[int, ActionLibraryEntry]:
    """Build a tiny deterministic action library by bounded random search."""

    entries = {}
    artifact_dir = Path(path).parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for offset, pitch in enumerate(midi_pitches):
        midi_path = artifact_dir / f"library_note_{pitch}.mid"
        write_monophonic_midi([NoteEvent(pitch, 0.0, 1.0, 90)], midi_path)
        env = ALAOneHandEnv(midi_path)
        env.reset()
        target_key = pitch - 21
        candidates = generate_random_candidates(
            env,
            count=candidate_count,
            seed=seed + offset,
        )
        best = None
        best_action = None
        for candidate in candidates:
            result = _evaluate_for_target(
                midi_path,
                target_key,
                candidate.action,
                horizon_steps=horizon_steps,
            )
            if best is None or _score_entry(result) > _score_entry(best):
                best = result
                best_action = candidate.action
        if best is None or best_action is None:
            raise RuntimeError(f"No action candidates evaluated for MIDI {pitch}.")
        entries[pitch] = ActionLibraryEntry(
            midi_pitch=pitch,
            key_index=target_key,
            action=tuple(float(v) for v in best_action),
            max_target_key_state=best["max_target_key_state"],
            max_unintended_key_state=best["max_unintended_key_state"],
            pressed_keys=tuple(best["pressed_keys"]),
            outcome=best["outcome"],
            debug_return=best["debug_return"],
        )
    save_action_library(entries, path)
    return entries


def save_action_library(entries: dict[int, ActionLibraryEntry], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(key): asdict(value) for key, value in entries.items()}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_action_library(path: str | Path) -> dict[int, ActionLibraryEntry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(key): ActionLibraryEntry(
            midi_pitch=int(value["midi_pitch"]),
            key_index=int(value["key_index"]),
            action=tuple(float(v) for v in value["action"]),
            max_target_key_state=float(value["max_target_key_state"]),
            max_unintended_key_state=float(value["max_unintended_key_state"]),
            pressed_keys=tuple(int(v) for v in value["pressed_keys"]),
            outcome=str(value["outcome"]),
            debug_return=float(value["debug_return"]),
        )
        for key, value in raw.items()
    }


def _evaluate_for_target(
    midi_path: Path,
    target_key: int,
    action: tuple[float, ...],
    *,
    horizon_steps: int,
) -> dict:
    env = ALAOneHandEnv(midi_path)
    env.reset()
    target_action = np.asarray(action, dtype=env.action_spec().dtype)
    max_target = 0.0
    max_unintended = 0.0
    pressed = set()
    native_rewards = []
    for step in range(horizon_steps):
        ramp = min(1.0, (step + 1) / 6.0)
        timestep = env.step(target_action * ramp)
        max_target = max(max_target, env.target_key_state(target_key) or 0.0)
        max_unintended = max(max_unintended, env.max_unintended_key_state(target_key))
        pressed.update(env.current_pressed_keys())
        if env.current_reward() is not None:
            native_rewards.append(float(env.current_reward()))
        if timestep.last():
            break
    wrong = len([key for key in pressed if key != target_key])
    clean = pressed == {target_key}
    near = max_target >= 0.25 and max_unintended <= max_target + 0.02
    outcome = _outcome(target_key, pressed, max_target, max_unintended)
    debug_return = 5.0 * max_target - 2.0 * max_unintended + (5.0 if clean else 0.0) + (2.0 if near else 0.0) - wrong
    return {
        "max_target_key_state": float(max_target),
        "max_unintended_key_state": float(max_unintended),
        "pressed_keys": tuple(sorted(pressed)),
        "outcome": outcome,
        "debug_return": float(debug_return),
        "native_reward_sum": float(sum(native_rewards)),
    }


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


def _score_entry(result: dict) -> tuple[float, float, float]:
    return (
        float(result["debug_return"]),
        float(result["max_target_key_state"]),
        -float(result["max_unintended_key_state"]),
    )
