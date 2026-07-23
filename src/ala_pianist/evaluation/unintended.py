"""Utilities for classifying unintended piano-key motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class UnintendedKeyClassification:
    """Summary of one key's relationship to the current target schedule."""

    key_index: int
    value: float
    category: str
    is_pressed: bool
    is_neighbouring_key: bool


def classify_unintended_key(
    key_index: int,
    *,
    value: float,
    current_target_keys: Iterable[int],
    previous_target_keys: Iterable[int],
    future_target_keys: Iterable[int],
    press_threshold: float,
) -> UnintendedKeyClassification:
    """Classify a non-current-target key state for diagnostics and rewards."""

    current = set(int(key) for key in current_target_keys)
    previous = set(int(key) for key in previous_target_keys) - current
    future = set(int(key) for key in future_target_keys) - current
    key = int(key_index)
    neighbours = {candidate for target in current for candidate in (target - 1, target + 1)}
    if key in previous:
        category = "previous_note_late_release"
    elif key in future:
        category = "future_note_early_activation"
    elif key in neighbours:
        category = "neighbouring_key_displacement"
    else:
        category = "unrelated_key_activation"
    return UnintendedKeyClassification(
        key_index=key,
        value=float(value),
        category=category,
        is_pressed=float(value) >= float(press_threshold),
        is_neighbouring_key=key in neighbours,
    )


def classify_unintended_keys(
    key_states: np.ndarray,
    *,
    current_target_keys: Iterable[int],
    previous_target_keys: Iterable[int],
    future_target_keys: Iterable[int],
    press_threshold: float,
) -> list[UnintendedKeyClassification]:
    """Classify every key that is not a current target."""

    current = set(int(key) for key in current_target_keys)
    result = []
    for key, value in enumerate(np.asarray(key_states, dtype=float)):
        if key in current:
            continue
        result.append(
            classify_unintended_key(
                key,
                value=float(value),
                current_target_keys=current,
                previous_target_keys=previous_target_keys,
                future_target_keys=future_target_keys,
                press_threshold=press_threshold,
            )
        )
    return result


def unintended_penalty_components(
    key_states: np.ndarray,
    *,
    current_target_keys: Iterable[int],
    previous_target_keys: Iterable[int],
    future_target_keys: Iterable[int],
    soft_threshold: float,
    press_threshold: float,
) -> dict[str, float]:
    """Compute scalar unintended-key components for reward shaping."""

    classifications = classify_unintended_keys(
        key_states,
        current_target_keys=current_target_keys,
        previous_target_keys=previous_target_keys,
        future_target_keys=future_target_keys,
        press_threshold=press_threshold,
    )
    soft = float(soft_threshold)
    press = float(press_threshold)
    span = max(1e-6, press - soft)

    continuous = 0.0
    barrier = 0.0
    unrelated_pressed = 0.0
    late_release = 0.0
    early_activation = 0.0
    duration = 0.0
    for item in classifications:
        excess = max(0.0, item.value - soft)
        duration += excess
        if item.category == "unrelated_key_activation":
            continuous += excess
            unrelated_pressed += 1.0 if item.is_pressed else 0.0
        if item.category == "previous_note_late_release":
            late_release += excess
        if item.category == "future_note_early_activation":
            early_activation += excess
        barrier += max(0.0, excess / span) ** 2
    return {
        "unintended_continuous_travel": continuous,
        "unintended_near_press_barrier": barrier,
        "unintended_pressed_event_count": unrelated_pressed,
        "late_release_travel": late_release,
        "early_activation_travel": early_activation,
        "unintended_integrated_duration": duration,
    }
