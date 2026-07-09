"""Bounded scripted diagnostic controller for generated monophonic MIDI clips."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ala_pianist.envs import ALAOneHandEnv


_FINGER_ACTUATORS = {
    "thumb": ("rh_A_THJ4", "rh_A_THJ2", "rh_A_THJ1"),
    "index": ("rh_A_FFJ3", "rh_A_FFJ0"),
    "middle": ("rh_A_MFJ3", "rh_A_MFJ0"),
    "ring": ("rh_A_RFJ3", "rh_A_RFJ0"),
    "little": ("rh_A_LFJ3", "rh_A_LFJ0"),
}

_LOCAL_KEY_TO_FINGER = {
    48: "thumb",
    49: "thumb",
    50: "index",
    51: "index",
    52: "middle",
    53: "ring",
    54: "little",
}


@dataclass(frozen=True)
class ScriptedStepLog:
    """One diagnostic control step."""

    step: int
    target_keys: tuple[int, ...]
    selected_key: int | None
    selected_finger: str | None
    pressed_keys: tuple[int, ...]
    reward: float | None
    discount: float | None
    last: bool
    status: str
    active_action_names: tuple[str, ...]


class ScriptedDiagnosticController:
    """Crude symbolic-target-to-action diagnostic.

    The controller is intentionally small and deterministic. It uses the public
    22D wrapper action, leaves sustain hidden, and makes a rough single-note
    press attempt by adjusting one finger group plus the default forearm slides.
    """

    def __init__(
        self,
        *,
        local_key_range: tuple[int, int] = (48, 54),
        forearm_tx_fraction_span: tuple[float, float] = (0.35, 0.65),
        forearm_ty_fraction: float = 0.65,
        finger_flexion_fraction: float = 0.85,
    ) -> None:
        self.local_key_range = local_key_range
        self.forearm_tx_fraction_span = forearm_tx_fraction_span
        self.forearm_ty_fraction = forearm_ty_fraction
        self.finger_flexion_fraction = finger_flexion_fraction

    def action(self, env: ALAOneHandEnv) -> tuple[np.ndarray, tuple[str, ...], int | None, str | None]:
        spec = env.action_spec()
        names = env.action_names()
        action = np.clip(np.zeros(spec.shape, dtype=spec.dtype), spec.minimum, spec.maximum)
        target_keys = env.current_target_keys()
        if not target_keys:
            return action, (), None, None

        selected_key = int(target_keys[0])
        selected_finger = self._finger_for_key(selected_key)
        active_names: list[str] = []

        self._set_forearm_targets(action, spec, names, selected_key, active_names)
        for actuator_name in _FINGER_ACTUATORS[selected_finger]:
            if actuator_name in names:
                idx = names.index(actuator_name)
                self._set_fraction(action, spec, idx, self.finger_flexion_fraction)
                active_names.append(actuator_name)

        return action, tuple(active_names), selected_key, selected_finger

    def _finger_for_key(self, key: int) -> str:
        if key in _LOCAL_KEY_TO_FINGER:
            return _LOCAL_KEY_TO_FINGER[key]
        lower, upper = self.local_key_range
        if upper <= lower:
            return "middle"
        fraction = min(1.0, max(0.0, (key - lower) / (upper - lower)))
        fingers = ("thumb", "index", "middle", "ring", "little")
        return fingers[min(len(fingers) - 1, int(round(fraction * (len(fingers) - 1))))]

    def _set_forearm_targets(
        self,
        action: np.ndarray,
        spec,
        names: tuple[str, ...],
        key: int,
        active_names: list[str],
    ) -> None:
        if "forearm_tx" in names:
            lower_key, upper_key = self.local_key_range
            key_fraction = 0.5
            if upper_key > lower_key:
                key_fraction = min(1.0, max(0.0, (key - lower_key) / (upper_key - lower_key)))
            low_frac, high_frac = self.forearm_tx_fraction_span
            idx = names.index("forearm_tx")
            self._set_fraction(action, spec, idx, low_frac + key_fraction * (high_frac - low_frac))
            active_names.append("forearm_tx")
        if "forearm_ty" in names:
            idx = names.index("forearm_ty")
            self._set_fraction(action, spec, idx, self.forearm_ty_fraction)
            active_names.append("forearm_ty")

    @staticmethod
    def _set_fraction(action: np.ndarray, spec, idx: int, fraction: float) -> None:
        fraction = min(1.0, max(0.0, fraction))
        action[idx] = spec.minimum[idx] + fraction * (spec.maximum[idx] - spec.minimum[idx])


def run_scripted_diagnostic(
    env: ALAOneHandEnv,
    controller: ScriptedDiagnosticController,
    *,
    max_steps: int = 64,
) -> list[ScriptedStepLog]:
    """Run a bounded diagnostic rollout and return structured logs."""

    timestep = env.reset()
    logs: list[ScriptedStepLog] = []

    for step in range(max_steps):
        target_keys = tuple(env.current_target_keys())
        action, active_names, selected_key, selected_finger = controller.action(env)
        timestep = env.step(action)
        pressed_keys = tuple(env.current_pressed_keys())
        logs.append(
            ScriptedStepLog(
                step=step,
                target_keys=target_keys,
                selected_key=selected_key,
                selected_finger=selected_finger,
                pressed_keys=pressed_keys,
                reward=env.current_reward(),
                discount=timestep.discount,
                last=timestep.last(),
                status=_classify_step(selected_key, pressed_keys),
                active_action_names=active_names,
            )
        )
        if timestep.last():
            break

    return logs


def _classify_step(selected_key: int | None, pressed_keys: tuple[int, ...]) -> str:
    if selected_key is None:
        return "no_target"
    if selected_key in pressed_keys and len(pressed_keys) == 1:
        return "clean_target_press"
    if selected_key in pressed_keys:
        return "target_press_with_unintended_keys"
    if pressed_keys:
        return "unintended_key_press"
    return "no_key_press"
