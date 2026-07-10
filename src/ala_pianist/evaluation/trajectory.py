"""Lightweight trajectory/demo recording for future MIDI-conditioned policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ala_pianist.envs import ALAOneHandEnv


@dataclass(frozen=True)
class TrajectoryRecord:
    timestep: int
    target_midi: int
    target_key: int
    compact_observation: tuple[float, ...]
    action: tuple[float, ...]
    target_key_state: float
    max_unintended_key_state: float
    pressed_keys: tuple[int, ...]
    closest_finger_to_target: str
    finger_target_distance: float | None
    target_contact_finger: str
    contact_pairs: tuple[str, ...]
    outcome: str


def record_action_rollout(
    env: ALAOneHandEnv,
    *,
    target_midi: int,
    action,
    horizon_steps: int = 24,
    ramp: bool = True,
) -> list[TrajectoryRecord]:
    """Run an action pattern and record compact per-step demo fields."""

    target_key = int(target_midi) - 21
    action = np.asarray(action, dtype=env.action_spec().dtype)
    records = []
    for step in range(horizon_steps):
        applied_action = action * min(1.0, (step + 1) / 6.0) if ramp else action
        timestep = env.step(applied_action)
        target_state = env.target_key_state(target_key) or 0.0
        max_unintended = env.max_unintended_key_state(target_key)
        pressed = tuple(env.current_pressed_keys())
        closest = env.nearest_fingertip_to_key(target_key)
        closest_name = str(closest["fingertip"]) if closest is not None else "unknown"
        closest_distance = float(closest["distance"]) if closest is not None else None
        contacts = tuple(f"{left} <-> {right}" for left, right in env.key_contact_pairs(target_key))
        contact_finger = _contact_finger(contacts)
        records.append(
            TrajectoryRecord(
                timestep=step,
                target_midi=int(target_midi),
                target_key=target_key,
                compact_observation=_compact_observation(
                    target_key,
                    target_state,
                    max_unintended,
                    len(pressed),
                    closest_distance,
                    step,
                    horizon_steps,
                ),
                action=tuple(float(v) for v in applied_action),
                target_key_state=float(target_state),
                max_unintended_key_state=float(max_unintended),
                pressed_keys=pressed,
                closest_finger_to_target=closest_name,
                finger_target_distance=closest_distance,
                target_contact_finger=contact_finger,
                contact_pairs=contacts,
                outcome=_outcome(target_key, set(pressed), target_state, max_unintended),
            )
        )
        if timestep.last():
            break
    return records


def save_trajectory_json(records: list[TrajectoryRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(record) for record in records], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _compact_observation(
    target_key: int,
    target_state: float,
    max_unintended: float,
    pressed_count: int,
    distance: float | None,
    step: int,
    horizon: int,
) -> tuple[float, ...]:
    return (
        float(target_key / 87.0),
        float(np.clip(target_state, 0.0, 1.0)),
        float(np.clip(max_unintended, 0.0, 1.0)),
        float(np.clip(pressed_count / 88.0, 0.0, 1.0)),
        float(1.0 if distance is None else min(1.0, distance / 0.20)),
        float(min(1.0, step / max(1, horizon))),
    )


def _outcome(target_key: int, pressed: set[int], target_state: float, max_unintended: float) -> str:
    if pressed == {target_key}:
        return "clean"
    if target_key in pressed:
        return "dirty"
    if target_state >= 0.25 and max_unintended <= target_state + 0.02:
        return "near_clean"
    if target_state > 0.02:
        return "partial"
    return "missed"


def _contact_finger(contact_pairs: tuple[str, ...]) -> str:
    text = " ".join(contact_pairs)
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
