"""Evaluation metrics for key-press rollouts."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class KeySetMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def to_dict(self, *, prefix: str = "") -> dict[str, float | int]:
        values = asdict(self)
        if prefix:
            return {f"{prefix}{key}": value for key, value in values.items()}
        return values


def pressed_key_metrics(
    target_keys: Iterable[int],
    pressed_keys: Iterable[int],
) -> KeySetMetrics:
    """Compute set-level precision/recall/F1 for keys pressed at least once."""

    target_set = {int(key) for key in target_keys}
    pressed_set = {int(key) for key in pressed_keys}
    tp = len(target_set & pressed_set)
    fp = len(pressed_set - target_set)
    fn = len(target_set - pressed_set)
    return _precision_recall_f1(tp, fp, fn)


def timestep_key_metrics(
    target_vectors: Iterable[Iterable[bool] | np.ndarray],
    pressed_vectors: Iterable[Iterable[bool] | np.ndarray],
) -> KeySetMetrics:
    """Compute aggregate timestep/key precision/recall/F1 over binary vectors."""

    target_array = np.asarray(list(target_vectors), dtype=bool)
    pressed_array = np.asarray(list(pressed_vectors), dtype=bool)
    if target_array.shape != pressed_array.shape:
        raise ValueError(
            "target_vectors and pressed_vectors must have the same shape: "
            f"{target_array.shape} != {pressed_array.shape}"
        )
    if target_array.size == 0:
        return _precision_recall_f1(0, 0, 0)
    tp = int(np.logical_and(target_array, pressed_array).sum())
    fp = int(np.logical_and(~target_array, pressed_array).sum())
    fn = int(np.logical_and(target_array, ~pressed_array).sum())
    return _precision_recall_f1(tp, fp, fn)


def binary_key_vector(keys: Iterable[int], *, n_keys: int = 88) -> np.ndarray:
    vector = np.zeros(n_keys, dtype=bool)
    for key in keys:
        key = int(key)
        if 0 <= key < n_keys:
            vector[key] = True
    return vector


def _precision_recall_f1(tp: int, fp: int, fn: int) -> KeySetMetrics:
    if tp == 0 and fp == 0 and fn == 0:
        return KeySetMetrics(tp, fp, fn, 1.0, 1.0, 1.0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return KeySetMetrics(tp, fp, fn, float(precision), float(recall), float(f1))
