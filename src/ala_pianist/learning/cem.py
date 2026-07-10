"""Tiny CEM-style single-note action refinement baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ala_pianist.baselines.calibration import TARGET_KEY, TARGET_MIDI, TARGET_NOTE
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.learning.random_search import (
    RandomSearchCandidate,
    RandomSearchResult,
    evaluate_action_pattern,
    evaluate_scripted_baseline,
    evaluate_zero_baseline,
    generate_random_candidates,
    write_single_note_learning_midi,
)


@dataclass(frozen=True)
class CEMResult:
    """One evaluated CEM candidate."""

    iteration: int
    candidate_index: int
    action: tuple[float, ...]
    metrics: RandomSearchResult


@dataclass(frozen=True)
class CEMIterationSummary:
    """Summary for one CEM update."""

    iteration: int
    best_debug_return: float
    best_target_key_state: float
    best_unintended_key_state: float
    best_outcome: str
    elite_count: int
    mean_action_abs: float
    std_action_mean: float


@dataclass(frozen=True)
class CEMSummary:
    """Full CEM diagnostic summary."""

    seed: int
    target_midi: int
    target_key: int
    target_note: str
    iterations: int
    candidate_count: int
    elite_fraction: float
    horizon_steps: int
    zero_result: RandomSearchResult
    scripted_result: RandomSearchResult
    random_search_result: RandomSearchResult
    best_result: CEMResult
    iteration_summaries: tuple[CEMIterationSummary, ...]


def generate_cem_candidates(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    count: int,
    rng: np.random.Generator,
    include_mean: bool = True,
) -> list[np.ndarray]:
    """Sample a deterministic bounded CEM candidate batch from the provided RNG."""

    candidates = []
    if include_mean:
        candidates.append(np.clip(mean, low, high).astype(float))
    while len(candidates) < count:
        sample = rng.normal(loc=mean, scale=std)
        candidates.append(np.clip(sample, low, high).astype(float))
    return candidates


def run_single_note_cem(
    midi_path: str | Path,
    *,
    iterations: int = 4,
    candidate_count: int = 20,
    elite_fraction: float = 0.25,
    horizon_steps: int = 24,
    seed: int = 13,
    random_search_seed: int = 7,
    random_search_candidates: int = 30,
    output_path: str | Path | None = None,
) -> CEMSummary:
    """Run a tiny CEM refinement over native public 22D action targets."""

    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if candidate_count <= 1:
        raise ValueError("candidate_count must be greater than 1.")
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_fraction must be in (0, 1].")

    zero_result = evaluate_zero_baseline(midi_path, horizon_steps=horizon_steps)
    scripted_result = evaluate_scripted_baseline(midi_path, horizon_steps=horizon_steps)

    template_env = ALAOneHandEnv(midi_path)
    template_env.reset()
    spec = template_env.action_spec()
    low = np.asarray(spec.minimum, dtype=float)
    high = np.asarray(spec.maximum, dtype=float)
    span = high - low

    random_candidates = generate_random_candidates(
        template_env,
        count=random_search_candidates,
        seed=random_search_seed,
    )
    random_results = [
        _evaluate_raw_action(midi_path, candidate.action, "random_seed", candidate.index, horizon_steps)
        for candidate in random_candidates
    ]
    best_random_result, best_random_candidate = max(
        zip(random_results, random_candidates),
        key=lambda item: _cem_score(item[0]),
    )

    mean = np.asarray(best_random_candidate.action, dtype=float)
    std = np.maximum(0.05 * span, 0.35 * np.abs(span))
    rng = np.random.default_rng(seed)
    elite_count = max(1, int(round(candidate_count * elite_fraction)))

    all_results: list[CEMResult] = []
    iteration_summaries = []
    best_cem: CEMResult | None = None
    for iteration in range(iterations):
        candidate_actions = generate_cem_candidates(
            mean=mean,
            std=std,
            low=low,
            high=high,
            count=candidate_count,
            rng=rng,
            include_mean=True,
        )
        iteration_results = []
        for candidate_index, action in enumerate(candidate_actions):
            metrics = _evaluate_raw_action(
                midi_path,
                tuple(action),
                "cem",
                candidate_index,
                horizon_steps,
            )
            result = CEMResult(
                iteration=iteration,
                candidate_index=candidate_index,
                action=tuple(float(v) for v in action),
                metrics=metrics,
            )
            iteration_results.append(result)
            all_results.append(result)
        iteration_results.sort(key=lambda result: _cem_score(result.metrics), reverse=True)
        elites = iteration_results[:elite_count]
        elite_actions = np.asarray([elite.action for elite in elites], dtype=float)
        mean = np.mean(elite_actions, axis=0)
        std = np.maximum(np.std(elite_actions, axis=0), 0.03 * span)
        std = np.minimum(std, 0.60 * span)

        if best_cem is None or _cem_score(iteration_results[0].metrics) > _cem_score(best_cem.metrics):
            best_cem = iteration_results[0]
        iteration_summaries.append(
            CEMIterationSummary(
                iteration=iteration,
                best_debug_return=iteration_results[0].metrics.debug_return,
                best_target_key_state=iteration_results[0].metrics.max_target_key_state,
                best_unintended_key_state=iteration_results[0].metrics.max_unintended_key_state,
                best_outcome=iteration_results[0].metrics.outcome,
                elite_count=elite_count,
                mean_action_abs=float(np.mean(np.abs(mean))),
                std_action_mean=float(np.mean(std)),
            )
        )

    if best_cem is None:
        raise RuntimeError("CEM did not evaluate any candidates.")
    summary = CEMSummary(
        seed=seed,
        target_midi=TARGET_MIDI,
        target_key=TARGET_KEY,
        target_note=TARGET_NOTE,
        iterations=iterations,
        candidate_count=candidate_count,
        elite_fraction=elite_fraction,
        horizon_steps=horizon_steps,
        zero_result=zero_result,
        scripted_result=scripted_result,
        random_search_result=best_random_result,
        best_result=best_cem,
        iteration_summaries=tuple(iteration_summaries),
    )
    if output_path is not None:
        write_cem_summary_json(summary, output_path)
    return summary


def write_cem_summary_json(summary: CEMSummary, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _evaluate_raw_action(
    midi_path: str | Path,
    action: tuple[float, ...],
    label: str,
    candidate_index: int,
    horizon_steps: int,
) -> RandomSearchResult:
    env = ALAOneHandEnv(midi_path)
    env.reset()
    return evaluate_action_pattern(
        env,
        np.asarray(action, dtype=env.action_spec().dtype),
        label=label,
        candidate_index=candidate_index,
        horizon_steps=horizon_steps,
    )


def _cem_score(result: RandomSearchResult) -> tuple[float, float, float, float]:
    clean_bonus = 10.0 if result.clean_target_press else 0.0
    near_bonus = 4.0 if result.near_clean_partial_press else 0.0
    target_press_bonus = 3.0 if TARGET_KEY in result.pressed_keys_seen else 0.0
    score = (
        8.0 * result.max_target_key_state
        - 4.0 * result.max_unintended_key_state
        - 2.0 * result.wrong_pressed_key_count
        + clean_bonus
        + near_bonus
        + target_press_bonus
        + (0.5 if result.any_target_contact else 0.0)
    )
    return (
        score,
        result.max_target_key_state,
        -result.max_unintended_key_state,
        -result.wrong_pressed_key_count,
    )


def write_single_note_cem_midi(path: str | Path) -> Path:
    return write_single_note_learning_midi(path)
