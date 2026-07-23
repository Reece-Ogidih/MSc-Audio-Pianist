"""Configurable staged curriculum schedules for Hex training runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CurriculumPhase:
    """One global-step interval in a staged curriculum."""

    name: str
    start_step: int
    end_step: int
    sequences: tuple[tuple[int, ...], ...]
    weights: tuple[float, ...]

    def contains(self, global_step: int) -> bool:
        return self.start_step <= int(global_step) < self.end_step

    @property
    def sequence_arg(self) -> str:
        return ";".join(",".join(str(pitch) for pitch in sequence) for sequence in self.sequences)

    @property
    def weight_arg(self) -> str:
        return ",".join(f"{weight:g}" for weight in self.weights)


def load_staged_curriculum(path: str | Path) -> tuple[CurriculumPhase, ...]:
    """Load and validate a staged curriculum JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    phases = []
    for item in payload.get("phases", ()):
        sequences = tuple(tuple(int(pitch) for pitch in sequence) for sequence in item["sequences"])
        weights = tuple(float(weight) for weight in item["weights"])
        phase = CurriculumPhase(
            name=str(item["name"]),
            start_step=int(item["start_step"]),
            end_step=int(item["end_step"]),
            sequences=sequences,
            weights=weights,
        )
        _validate_phase(phase)
        phases.append(phase)
    if not phases:
        raise ValueError("Staged curriculum must contain at least one phase.")
    phases = sorted(phases, key=lambda phase: phase.start_step)
    for previous, current in zip(phases, phases[1:]):
        if previous.end_step != current.start_step:
            raise ValueError(
                f"Curriculum phases must be contiguous: {previous.name} ends at "
                f"{previous.end_step}, {current.name} starts at {current.start_step}."
            )
    return tuple(phases)


def phase_for_global_step(phases: tuple[CurriculumPhase, ...], global_step: int) -> CurriculumPhase:
    """Return the phase active at a cumulative global training step."""

    for phase in phases:
        if phase.contains(global_step):
            return phase
    if global_step == phases[-1].end_step:
        return phases[-1]
    raise ValueError(f"No staged curriculum phase contains global step {global_step}.")


def _validate_phase(phase: CurriculumPhase) -> None:
    if phase.end_step <= phase.start_step:
        raise ValueError(f"Phase {phase.name!r} must have end_step > start_step.")
    if not phase.sequences:
        raise ValueError(f"Phase {phase.name!r} must include at least one sequence.")
    if len(phase.sequences) != len(phase.weights):
        raise ValueError(f"Phase {phase.name!r} sequence/weight length mismatch.")
    if any(not sequence for sequence in phase.sequences):
        raise ValueError(f"Phase {phase.name!r} contains an empty sequence.")
    if any(weight < 0.0 for weight in phase.weights):
        raise ValueError(f"Phase {phase.name!r} weights must be non-negative.")
    total = sum(phase.weights)
    if total <= 0.0:
        raise ValueError(f"Phase {phase.name!r} must contain at least one positive weight.")


def phases_as_dicts(phases: tuple[CurriculumPhase, ...]) -> list[dict[str, Any]]:
    """Serialise phases for metadata files."""

    return [
        {
            "name": phase.name,
            "start_step": phase.start_step,
            "end_step": phase.end_step,
            "sequences": [list(sequence) for sequence in phase.sequences],
            "weights": list(phase.weights),
            "sequence_arg": phase.sequence_arg,
            "weight_arg": phase.weight_arg,
        }
        for phase in phases
    ]
