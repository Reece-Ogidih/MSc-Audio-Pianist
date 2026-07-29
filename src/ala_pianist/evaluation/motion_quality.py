"""Motion-quality diagnostics for frozen rollout traces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ACTION_NAMES: tuple[str, ...] = (
    "rh_A_WRJ2",
    "rh_A_WRJ1",
    "rh_A_THJ5",
    "rh_A_THJ4",
    "rh_A_THJ3",
    "rh_A_THJ2",
    "rh_A_THJ1",
    "rh_A_FFJ4",
    "rh_A_FFJ3",
    "rh_A_FFJ0",
    "rh_A_MFJ4",
    "rh_A_MFJ3",
    "rh_A_MFJ0",
    "rh_A_RFJ4",
    "rh_A_RFJ3",
    "rh_A_RFJ0",
    "rh_A_LFJ5",
    "rh_A_LFJ4",
    "rh_A_LFJ3",
    "rh_A_LFJ0",
    "forearm_tx",
    "forearm_ty",
)


@dataclass(frozen=True)
class ActionDimension:
    index: int
    name: str
    group: str
    native_lower: float | None = None
    native_upper: float | None = None


def action_columns(columns: Iterable[str]) -> list[str]:
    return sorted(column for column in columns if column.startswith("action_"))


def fingertip_prefixes(columns: Iterable[str]) -> list[str]:
    suffix = "_x"
    prefixes = []
    for column in columns:
        if column.startswith("fingertip_") and column.endswith(suffix):
            prefix = column[: -len(suffix)]
            if f"{prefix}_y" in columns and f"{prefix}_z" in columns:
                prefixes.append(prefix)
    return sorted(prefixes)


def infer_action_group(name: str) -> str:
    if name in {"forearm_tx", "forearm_ty"}:
        return "forearm_translation"
    if "WRJ" in name:
        return "wrist"
    if "_TH" in name:
        return "thumb"
    if "_FF" in name:
        return "index_finger"
    if "_MF" in name:
        return "middle_finger"
    if "_RF" in name:
        return "ring_finger"
    if "_LF" in name:
        return "little_finger"
    return "unknown"


def action_dimension_mapping(
    native_bounds: list[tuple[float, float]] | None = None,
) -> list[ActionDimension]:
    bounds = native_bounds or [(None, None)] * len(ACTION_NAMES)
    return [
        ActionDimension(
            index=index,
            name=name,
            group=infer_action_group(name),
            native_lower=None if bounds[index][0] is None else float(bounds[index][0]),
            native_upper=None if bounds[index][1] is None else float(bounds[index][1]),
        )
        for index, name in enumerate(ACTION_NAMES)
    ]


def phase_labels(trace: pd.DataFrame) -> pd.Series:
    """Segment a rollout into coarse note phases using trace target markers.

    The traces already contain synchronized `intended_key_indices`, `onset_event`
    and `release_event` fields. This function derives deterministic phases from
    those existing columns without re-running the simulator.
    """

    n = len(trace)
    if n == 0:
        return pd.Series([], dtype=object)
    intended = [tuple(_parse_list(value)) for value in trace["intended_key_indices"]]
    nonempty = [idx for idx, keys in enumerate(intended) if keys]
    labels = np.full(n, "terminal_release", dtype=object)
    if not nonempty:
        return pd.Series(labels, index=trace.index)

    first_start = min(nonempty)
    labels[:first_start] = "approach_first_note"
    target_runs = _runs([(idx, intended[idx]) for idx in nonempty])
    if target_runs:
        start, end, _keys = target_runs[0]
        labels[start : end + 1] = "first_note_hold"
        if len(target_runs) == 1:
            labels[end + 1 :] = "terminal_release"
        else:
            next_start, next_end, _ = target_runs[1]
            labels[end + 1 : next_start] = "release_transition"
            labels[next_start : next_end + 1] = "second_note_hold"
            if next_end + 1 < n:
                labels[next_end + 1 :] = "terminal_release"
            for extra_start, extra_end, _ in target_runs[2:]:
                labels[extra_start : extra_end + 1] = "later_note_hold"
    return pd.Series(labels, index=trace.index)


def _runs(indexed_keys: list[tuple[int, tuple[int, ...]]]) -> list[tuple[int, int, tuple[int, ...]]]:
    if not indexed_keys:
        return []
    runs = []
    start, previous, keys = indexed_keys[0][0], indexed_keys[0][0], indexed_keys[0][1]
    for index, next_keys in indexed_keys[1:]:
        if index == previous + 1 and next_keys == keys:
            previous = index
            continue
        runs.append((start, previous, keys))
        start, previous, keys = index, index, next_keys
    runs.append((start, previous, keys))
    return runs


def _parse_list(value) -> list[int]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return [int(item) for item in parsed]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(item) for item in value]
    if pd.isna(value):
        return []
    return [int(value)]


def action_quality(actions: np.ndarray, *, saturation_threshold: float = 0.95) -> dict[str, float]:
    actions = np.asarray(actions, dtype=float)
    if actions.size == 0:
        return _empty_action_metrics()
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    delta = np.diff(actions, axis=0)
    sign_changes = np.diff(np.sign(actions), axis=0) != 0
    high_freq = np.zeros_like(actions, dtype=bool)
    if len(actions) >= 3:
        high_freq[2:] = np.sign(np.diff(actions, n=2, axis=0)) != 0
    return {
        "mean_abs_action": float(np.mean(np.abs(actions))),
        "action_std": float(np.std(actions)),
        "mean_abs_action_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "mean_squared_action_delta": float(np.mean(delta**2)) if delta.size else 0.0,
        "max_abs_action_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "action_saturation_fraction": saturation_fraction(actions, saturation_threshold),
        "sign_change_rate": float(np.mean(sign_changes)) if sign_changes.size else 0.0,
        "high_frequency_oscillation_rate": float(np.mean(high_freq)) if high_freq.size else 0.0,
    }


def per_dimension_action_quality(
    actions: np.ndarray,
    *,
    saturation_threshold: float = 0.95,
) -> list[dict[str, float]]:
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2:
        raise ValueError("Expected a 2D action array.")
    rows = []
    for index in range(actions.shape[1]):
        metrics = action_quality(actions[:, [index]], saturation_threshold=saturation_threshold)
        rows.append({"action_index": index, **metrics})
    return rows


def _empty_action_metrics() -> dict[str, float]:
    return {
        "mean_abs_action": 0.0,
        "action_std": 0.0,
        "mean_abs_action_delta": 0.0,
        "mean_squared_action_delta": 0.0,
        "max_abs_action_delta": 0.0,
        "action_saturation_fraction": 0.0,
        "sign_change_rate": 0.0,
        "high_frequency_oscillation_rate": 0.0,
    }


def saturation_fraction(values: np.ndarray, threshold: float = 0.95) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(np.abs(values) >= float(threshold)))


def fingertip_motion_quality(trace: pd.DataFrame, *, dt: float) -> dict[str, float]:
    prefixes = fingertip_prefixes(trace.columns)
    if not prefixes or len(trace) < 2:
        return {
            "mean_fingertip_displacement": 0.0,
            "p95_fingertip_displacement": 0.0,
            "max_fingertip_displacement": 0.0,
            "mean_fingertip_speed": 0.0,
            "p95_fingertip_speed": 0.0,
            "max_fingertip_speed": 0.0,
            "mean_fingertip_acceleration": 0.0,
            "p95_fingertip_acceleration": 0.0,
            "max_fingertip_acceleration": 0.0,
            "mean_fingertip_jerk": 0.0,
            "p95_fingertip_jerk": 0.0,
            "max_fingertip_jerk": 0.0,
        }
    positions = np.stack(
        [
            trace[[f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]].to_numpy(dtype=float)
            for prefix in prefixes
        ],
        axis=1,
    )
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    speed = displacement / dt
    velocity = np.diff(positions, axis=0) / dt
    acceleration_vec = np.diff(velocity, axis=0) / dt
    jerk_vec = np.diff(acceleration_vec, axis=0) / dt
    acceleration = np.linalg.norm(acceleration_vec, axis=2) if acceleration_vec.size else np.zeros(0)
    jerk = np.linalg.norm(jerk_vec, axis=2) if jerk_vec.size else np.zeros(0)
    return {
        "mean_fingertip_displacement": _mean(displacement),
        "p95_fingertip_displacement": _percentile(displacement, 95),
        "max_fingertip_displacement": _max(displacement),
        "mean_fingertip_speed": _mean(speed),
        "p95_fingertip_speed": _percentile(speed, 95),
        "max_fingertip_speed": _max(speed),
        "mean_fingertip_acceleration": _mean(acceleration),
        "p95_fingertip_acceleration": _percentile(acceleration, 95),
        "max_fingertip_acceleration": _max(acceleration),
        "mean_fingertip_jerk": _mean(jerk),
        "p95_fingertip_jerk": _percentile(jerk, 95),
        "max_fingertip_jerk": _max(jerk),
    }


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if np.asarray(values).size else 0.0


def _max(values: np.ndarray) -> float:
    return float(np.max(values)) if np.asarray(values).size else 0.0


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if np.asarray(values).size else 0.0


def standardized_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    pooled = sqrt((float(np.var(a, ddof=0)) + float(np.var(b, ddof=0))) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int = 13,
    samples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=values.size, replace=True))) for _ in range(samples)]
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def annotate_rollout_frame(
    frame: np.ndarray,
    *,
    sequence_id: str,
    simulation_time: float,
    intended_midi: list[int],
    intended_keys: list[int],
    pressed_midi: list[int],
    wrong_pressed_midi: list[int],
    phase: str,
    failure_labels: list[str],
    target_activation: float,
    max_unintended: float,
    key_range: tuple[int, int] = (72, 76),
) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, _height = image.size
    draw.rectangle((0, 0, width, 104), fill=(0, 0, 0, 150))
    lines = [
        f"{sequence_id}  t={simulation_time:.2f}s  phase={phase}",
        f"target MIDI={intended_midi or []} key={intended_keys or []}  pressed MIDI={pressed_midi or []}",
        f"wrong MIDI={wrong_pressed_midi or []}  target={target_activation:.3f} unintended={max_unintended:.3f}",
        "labels=" + ", ".join(failure_labels[:4]),
    ]
    for idx, line in enumerate(lines):
        draw.text((8, 7 + idx * 22), line, fill=(255, 255, 255, 255))
    x0, y0, key_w, key_h = 12, 112, 38, 34
    lo, hi = key_range
    for offset, midi in enumerate(range(lo, hi + 1)):
        x = x0 + offset * key_w
        is_black = midi % 12 in {1, 3, 6, 8, 10}
        fill = (35, 35, 35, 210) if is_black else (245, 245, 245, 210)
        outline = (255, 210, 80, 255) if midi in intended_midi else (60, 60, 60, 255)
        draw.rectangle((x, y0, x + key_w - 4, y0 + key_h), fill=fill, outline=outline, width=2)
        text_fill = (255, 255, 255, 255) if is_black else (20, 20, 20, 255)
        draw.text((x + 6, y0 + 9), str(midi), fill=text_fill)
    return np.asarray(image)

