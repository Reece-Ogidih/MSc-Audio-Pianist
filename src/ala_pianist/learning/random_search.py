"""Small seeded random-search debug loop for a generated single-note task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ala_pianist.baselines import ScriptedDiagnosticController, run_scripted_diagnostic
from ala_pianist.baselines.calibration import TARGET_KEY, TARGET_MIDI, TARGET_NOTE
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


@dataclass(frozen=True)
class RandomSearchCandidate:
    """One open-loop action pattern candidate."""

    index: int
    action: tuple[float, ...]


@dataclass(frozen=True)
class RandomSearchResult:
    """Metrics for one evaluated action pattern."""

    label: str
    candidate_index: int | None
    target_midi: int
    target_key: int
    target_note: str
    horizon_steps: int
    debug_return: float
    native_reward_sum: float
    max_target_key_state: float
    max_unintended_key_state: float
    min_fingertip_target_distance: float | None
    any_target_contact: bool
    any_key_contact: bool
    clean_target_press: bool
    near_clean_partial_press: bool
    wrong_pressed_key_count: int
    pressed_keys_seen: tuple[int, ...]
    outcome: str


@dataclass(frozen=True)
class RandomSearchSummary:
    """Summary of a tiny random-search debug run."""

    seed: int
    target_midi: int
    target_key: int
    target_note: str
    candidate_count: int
    horizon_steps: int
    zero_result: RandomSearchResult
    scripted_result: RandomSearchResult
    best_result: RandomSearchResult
    results: tuple[RandomSearchResult, ...]


def write_single_note_learning_midi(path: str | Path) -> Path:
    """Write the generated one-note D#5 clip used by the debug learner."""

    return write_monophonic_midi(
        [NoteEvent(TARGET_MIDI, 0.0, 1.0, 90)],
        path,
        title="single note random search D sharp 5",
    )


def generate_random_candidates(
    env: ALAOneHandEnv,
    *,
    count: int,
    seed: int,
    scale: float = 0.75,
) -> list[RandomSearchCandidate]:
    """Generate deterministic bounded 22D target actions."""

    rng = np.random.default_rng(seed)
    spec = env.action_spec()
    low = np.asarray(spec.minimum, dtype=float)
    high = np.asarray(spec.maximum, dtype=float)
    center = np.clip(np.zeros(spec.shape, dtype=float), low, high)
    span_low = center - scale * (center - low)
    span_high = center + scale * (high - center)
    candidates = []
    for index in range(count):
        action = rng.uniform(span_low, span_high).astype(float)
        candidates.append(RandomSearchCandidate(index=index, action=tuple(action)))
    return candidates


def run_single_note_random_search(
    midi_path: str | Path,
    *,
    candidate_count: int = 30,
    horizon_steps: int = 24,
    seed: int = 7,
    output_path: str | Path | None = None,
) -> RandomSearchSummary:
    """Compare zero/scripted baselines with a tiny seeded random search."""

    zero_result = evaluate_zero_baseline(midi_path, horizon_steps=horizon_steps)
    scripted_result = evaluate_scripted_baseline(midi_path, horizon_steps=horizon_steps)

    template_env = ALAOneHandEnv(midi_path)
    template_env.reset()
    candidates = generate_random_candidates(template_env, count=candidate_count, seed=seed)

    results = []
    for candidate in candidates:
        env = ALAOneHandEnv(midi_path)
        env.reset()
        result = evaluate_action_pattern(
            env,
            np.asarray(candidate.action, dtype=env.action_spec().dtype),
            label="random_search",
            candidate_index=candidate.index,
            horizon_steps=horizon_steps,
        )
        results.append(result)

    best = max(results, key=_score_result) if results else zero_result
    summary = RandomSearchSummary(
        seed=seed,
        target_midi=TARGET_MIDI,
        target_key=TARGET_KEY,
        target_note=TARGET_NOTE,
        candidate_count=candidate_count,
        horizon_steps=horizon_steps,
        zero_result=zero_result,
        scripted_result=scripted_result,
        best_result=best,
        results=tuple(results),
    )
    if output_path is not None:
        write_summary_json(summary, output_path)
    return summary


def evaluate_zero_baseline(
    midi_path: str | Path,
    *,
    horizon_steps: int,
) -> RandomSearchResult:
    env = ALAOneHandEnv(midi_path)
    env.reset()
    action = np.zeros(env.action_spec().shape, dtype=env.action_spec().dtype)
    return evaluate_action_pattern(
        env,
        action,
        label="zero",
        candidate_index=None,
        horizon_steps=horizon_steps,
    )


def evaluate_scripted_baseline(
    midi_path: str | Path,
    *,
    horizon_steps: int,
) -> RandomSearchResult:
    env = ALAOneHandEnv(midi_path)
    logs = run_scripted_diagnostic(
        env,
        ScriptedDiagnosticController(),
        max_steps=horizon_steps,
    )
    max_target = max((log.target_key_state or 0.0) for log in logs) if logs else 0.0
    max_unintended = max((log.max_unintended_key_state for log in logs), default=0.0)
    distances = [
        log.nearest_fingertip_distance
        for log in logs
        if log.nearest_fingertip_distance is not None
    ]
    pressed = sorted({key for log in logs for key in log.pressed_keys})
    native_reward_sum = sum(float(log.reward) for log in logs if log.reward is not None)
    any_target_contact = any(log.target_contact for log in logs)
    any_key_contact = any(log.any_key_contact for log in logs)
    clean = TARGET_KEY in pressed and pressed == [TARGET_KEY]
    near_clean = max_target >= 0.25 and max_unintended <= max_target + 0.02
    wrong_count = len([key for key in pressed if key != TARGET_KEY])
    debug_return = _debug_score(
        max_target_state=max_target,
        max_unintended_state=max_unintended,
        clean_target_press=clean,
        near_clean_partial_press=near_clean,
        wrong_pressed_key_count=wrong_count,
        any_target_contact=any_target_contact,
    )
    return RandomSearchResult(
        label="scripted",
        candidate_index=None,
        target_midi=TARGET_MIDI,
        target_key=TARGET_KEY,
        target_note=TARGET_NOTE,
        horizon_steps=horizon_steps,
        debug_return=debug_return,
        native_reward_sum=native_reward_sum,
        max_target_key_state=max_target,
        max_unintended_key_state=max_unintended,
        min_fingertip_target_distance=min(distances) if distances else None,
        any_target_contact=any_target_contact,
        any_key_contact=any_key_contact,
        clean_target_press=clean,
        near_clean_partial_press=near_clean,
        wrong_pressed_key_count=wrong_count,
        pressed_keys_seen=tuple(pressed),
        outcome=_classify_outcome(clean, near_clean, any_target_contact, max_target, pressed),
    )


def evaluate_action_pattern(
    env: ALAOneHandEnv,
    target_action: np.ndarray,
    *,
    label: str,
    candidate_index: int | None,
    horizon_steps: int,
) -> RandomSearchResult:
    """Evaluate one ramped open-loop 22D action pattern."""

    target_action = np.asarray(target_action, dtype=env.action_spec().dtype)
    max_target_state = 0.0
    max_unintended_state = 0.0
    min_distance = None
    any_target_contact = False
    any_key_contact = False
    pressed_keys_seen: set[int] = set()
    native_rewards = []

    for step in range(horizon_steps):
        ramp = min(1.0, (step + 1) / 6.0)
        timestep = env.step(target_action * ramp)
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
        pressed_keys_seen.update(env.current_pressed_keys())
        if env.current_reward() is not None:
            native_rewards.append(float(env.current_reward()))
        if timestep.last():
            break

    pressed = tuple(sorted(pressed_keys_seen))
    clean = TARGET_KEY in pressed_keys_seen and pressed_keys_seen == {TARGET_KEY}
    near_clean = max_target_state >= 0.25 and max_unintended_state <= max_target_state + 0.02
    wrong_count = len([key for key in pressed_keys_seen if key != TARGET_KEY])
    action_penalty = 0.002 * float(np.mean(np.square(target_action)))
    debug_return = _debug_score(
        max_target_state=max_target_state,
        max_unintended_state=max_unintended_state,
        clean_target_press=clean,
        near_clean_partial_press=near_clean,
        wrong_pressed_key_count=wrong_count,
        any_target_contact=any_target_contact,
    ) - action_penalty
    return RandomSearchResult(
        label=label,
        candidate_index=candidate_index,
        target_midi=TARGET_MIDI,
        target_key=TARGET_KEY,
        target_note=TARGET_NOTE,
        horizon_steps=horizon_steps,
        debug_return=debug_return,
        native_reward_sum=sum(native_rewards),
        max_target_key_state=max_target_state,
        max_unintended_key_state=max_unintended_state,
        min_fingertip_target_distance=min_distance,
        any_target_contact=any_target_contact,
        any_key_contact=any_key_contact,
        clean_target_press=clean,
        near_clean_partial_press=near_clean,
        wrong_pressed_key_count=wrong_count,
        pressed_keys_seen=pressed,
        outcome=_classify_outcome(
            clean,
            near_clean,
            any_target_contact,
            max_target_state,
            pressed,
        ),
    )


def write_summary_json(summary: RandomSearchSummary, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _debug_score(
    *,
    max_target_state: float,
    max_unintended_state: float,
    clean_target_press: bool,
    near_clean_partial_press: bool,
    wrong_pressed_key_count: int,
    any_target_contact: bool,
) -> float:
    return (
        5.0 * max_target_state
        - 2.0 * max_unintended_state
        + (5.0 if clean_target_press else 0.0)
        + (2.0 if near_clean_partial_press else 0.0)
        + (0.5 if any_target_contact else 0.0)
        - float(wrong_pressed_key_count)
    )


def _classify_outcome(
    clean: bool,
    near_clean: bool,
    any_target_contact: bool,
    max_target_state: float,
    pressed: tuple[int, ...] | list[int],
) -> str:
    if clean:
        return "clean_target_press"
    if TARGET_KEY in pressed:
        return "target_press_with_unintended_keys"
    if near_clean:
        return "near_clean_partial_press"
    if any_target_contact or max_target_state > 0.02:
        return "contact_without_sufficient_travel"
    return "no_target_progress"


def _score_result(result: RandomSearchResult) -> tuple[float, float, float]:
    return (
        result.debug_return,
        result.max_target_key_state,
        -result.max_unintended_key_state,
    )
