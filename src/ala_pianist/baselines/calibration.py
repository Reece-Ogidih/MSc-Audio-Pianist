"""Bounded single-note calibration sweeps for scripted diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


TARGET_MIDI = 75
TARGET_KEY = TARGET_MIDI - 21
TARGET_NOTE = "D#5"
TARGET_FINGER = "little"
TARGET_FINGER_ACTUATORS = ("rh_A_LFJ3", "rh_A_LFJ0")

_NEAR_CLEAN_TRAVEL = 0.25
_KEY_TRAVEL_EPSILON = 0.02


@dataclass(frozen=True)
class CalibrationCandidate:
    """One deterministic action candidate for a single-note sweep."""

    index: int
    forearm_tx_fraction: float
    forearm_ty_fraction: float
    finger_flexion_fraction: float


@dataclass(frozen=True)
class CalibrationResult:
    """Summary of one bounded calibration candidate rollout."""

    candidate: CalibrationCandidate
    target_midi: int
    target_key: int
    target_note: str
    selected_finger: str
    horizon_steps: int
    max_target_key_state: float
    max_unintended_key_state: float
    min_fingertip_target_distance: float | None
    any_target_contact: bool
    any_key_contact: bool
    best_reward: float | None
    final_reward: float | None
    pressed_keys_seen: tuple[int, ...]
    contact_pairs: tuple[str, ...]
    outcome: str


def generate_single_note_candidates() -> list[CalibrationCandidate]:
    """Return a small deterministic grid around the prior D#5 partial-travel regime."""

    candidates = []
    index = 0
    for forearm_tx in (0.55, 0.65, 0.75):
        for forearm_ty in (0.55, 0.70):
            for finger_flexion in (0.75, 0.95):
                candidates.append(
                    CalibrationCandidate(
                        index=index,
                        forearm_tx_fraction=forearm_tx,
                        forearm_ty_fraction=forearm_ty,
                        finger_flexion_fraction=finger_flexion,
                    )
                )
                index += 1
    return candidates


def write_single_note_calibration_midi(path: str | Path) -> Path:
    """Write the generated one-note D#5 calibration clip."""

    return write_monophonic_midi(
        [NoteEvent(TARGET_MIDI, 0.0, 1.0, 90)],
        path,
        title="single note calibration D sharp 5",
    )


def run_single_note_calibration_sweep(
    midi_path: str | Path,
    *,
    candidates: list[CalibrationCandidate] | None = None,
    horizon_steps: int = 24,
) -> list[CalibrationResult]:
    """Run a bounded deterministic D#5 calibration sweep through the public wrapper."""

    candidates = candidates or generate_single_note_candidates()
    results = []
    for candidate in candidates:
        env = ALAOneHandEnv(midi_path)
        env.reset()
        action = _candidate_action(env, candidate)
        result = _run_candidate(env, candidate, action, horizon_steps=horizon_steps)
        results.append(result)
    return results


def best_calibration_result(results: list[CalibrationResult]) -> CalibrationResult:
    """Pick the most informative result, preferring clean presses then target travel."""

    if not results:
        raise ValueError("At least one calibration result is required.")
    return max(results, key=_result_score)


def _candidate_action(env: ALAOneHandEnv, candidate: CalibrationCandidate) -> np.ndarray:
    spec = env.action_spec()
    names = env.action_names()
    action = np.clip(np.zeros(spec.shape, dtype=spec.dtype), spec.minimum, spec.maximum)

    _set_named_fraction(action, spec, names, "forearm_tx", candidate.forearm_tx_fraction)
    _set_named_fraction(action, spec, names, "forearm_ty", candidate.forearm_ty_fraction)
    for actuator_name in TARGET_FINGER_ACTUATORS:
        _set_named_fraction(
            action,
            spec,
            names,
            actuator_name,
            candidate.finger_flexion_fraction,
        )
    return action


def _run_candidate(
    env: ALAOneHandEnv,
    candidate: CalibrationCandidate,
    action: np.ndarray,
    *,
    horizon_steps: int,
) -> CalibrationResult:
    max_target_state = 0.0
    max_unintended_state = 0.0
    min_distance = None
    any_target_contact = False
    any_key_contact = False
    rewards: list[float] = []
    pressed_keys_seen: set[int] = set()
    contact_pairs: set[str] = set()
    final_reward = None

    for step in range(horizon_steps):
        ramp = min(1.0, (step + 1) / 6.0)
        timestep = env.step(action * ramp)
        target_state = env.target_key_state(TARGET_KEY) or 0.0
        unintended_state = env.max_unintended_key_state(TARGET_KEY)
        nearest = env.nearest_fingertip_to_key(TARGET_KEY)
        target_contacts = env.key_contact_pairs(TARGET_KEY)
        key_contacts = env.key_contact_pairs(None)

        max_target_state = max(max_target_state, float(target_state))
        max_unintended_state = max(max_unintended_state, float(unintended_state))
        if nearest is not None:
            distance = float(nearest["distance"])
            min_distance = distance if min_distance is None else min(min_distance, distance)
        any_target_contact = any_target_contact or bool(target_contacts)
        any_key_contact = any_key_contact or bool(key_contacts)
        for pair in key_contacts:
            contact_pairs.add(f"{pair[0]} <-> {pair[1]}")
        pressed_keys_seen.update(env.current_pressed_keys())
        final_reward = env.current_reward()
        if final_reward is not None:
            rewards.append(float(final_reward))
        if timestep.last():
            break

    outcome = _classify_result(
        target_key=TARGET_KEY,
        pressed_keys_seen=pressed_keys_seen,
        max_target_state=max_target_state,
        max_unintended_state=max_unintended_state,
        any_target_contact=any_target_contact,
        any_key_contact=any_key_contact,
        min_distance=min_distance,
    )
    return CalibrationResult(
        candidate=candidate,
        target_midi=TARGET_MIDI,
        target_key=TARGET_KEY,
        target_note=TARGET_NOTE,
        selected_finger=TARGET_FINGER,
        horizon_steps=horizon_steps,
        max_target_key_state=max_target_state,
        max_unintended_key_state=max_unintended_state,
        min_fingertip_target_distance=min_distance,
        any_target_contact=any_target_contact,
        any_key_contact=any_key_contact,
        best_reward=max(rewards) if rewards else None,
        final_reward=final_reward,
        pressed_keys_seen=tuple(sorted(pressed_keys_seen)),
        contact_pairs=tuple(sorted(contact_pairs)),
        outcome=outcome,
    )


def _classify_result(
    *,
    target_key: int,
    pressed_keys_seen: set[int],
    max_target_state: float,
    max_unintended_state: float,
    any_target_contact: bool,
    any_key_contact: bool,
    min_distance: float | None,
) -> str:
    if target_key in pressed_keys_seen and pressed_keys_seen == {target_key}:
        return "clean_target_press"
    if target_key in pressed_keys_seen:
        return "target_press_with_unintended_keys"
    if (
        max_target_state >= _NEAR_CLEAN_TRAVEL
        and max_unintended_state <= max_target_state + _KEY_TRAVEL_EPSILON
    ):
        return "near_clean_partial_press"
    if any_target_contact or max_target_state > _KEY_TRAVEL_EPSILON:
        return "contact_without_sufficient_travel"
    if any_key_contact or max_unintended_state > _KEY_TRAVEL_EPSILON:
        return "unstable_or_invalid"
    if min_distance is None or min_distance > 0.035:
        return "no_contact_or_no_approach"
    return "unstable_or_invalid"


def _set_named_fraction(
    action: np.ndarray,
    spec,
    names: tuple[str, ...],
    action_name: str,
    fraction: float,
) -> None:
    if action_name not in names:
        raise ValueError(f"Action name {action_name!r} is not available.")
    fraction = min(1.0, max(0.0, fraction))
    idx = names.index(action_name)
    action[idx] = spec.minimum[idx] + fraction * (spec.maximum[idx] - spec.minimum[idx])


def _result_score(result: CalibrationResult) -> tuple[int, float, float, float]:
    ranks = {
        "clean_target_press": 5,
        "near_clean_partial_press": 4,
        "target_press_with_unintended_keys": 3,
        "contact_without_sufficient_travel": 2,
        "no_contact_or_no_approach": 1,
        "unstable_or_invalid": 0,
    }
    distance = result.min_fingertip_target_distance
    return (
        ranks.get(result.outcome, 0),
        result.max_target_key_state,
        -result.max_unintended_key_state,
        -(distance if distance is not None else 1.0),
    )
