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

_APPROACH_DISTANCE_METERS = 0.035
_KEY_TRAVEL_EPSILON = 0.02


@dataclass(frozen=True)
class ScriptedStepLog:
    """One diagnostic control step."""

    step: int
    target_keys: tuple[int, ...]
    selected_key: int | None
    selected_note: str | None
    selected_finger: str | None
    pressed_keys: tuple[int, ...]
    target_key_state: float | None
    max_unintended_key_state: float
    nearest_fingertip: str | None
    nearest_fingertip_distance: float | None
    target_contact: bool
    any_key_contact: bool
    wrong_key_nearest_distance: float | None
    contact_pairs: tuple[str, ...]
    reward: float | None
    discount: float | None
    last: bool
    status: str
    diagnostic_category: str
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
        target_state = env.target_key_state(selected_key)
        max_unintended_state = env.max_unintended_key_state(selected_key)
        nearest = env.nearest_fingertip_to_key(selected_key)
        target_contacts = [] if selected_key is None else env.key_contact_pairs(selected_key)
        any_key_contacts = env.key_contact_pairs(None)
        wrong_key_distance = _nearest_wrong_key_distance(env, selected_key, pressed_keys)
        diagnostic_category = _classify_step(
            selected_key=selected_key,
            pressed_keys=pressed_keys,
            target_key_state=target_state,
            max_unintended_key_state=max_unintended_state,
            nearest_fingertip_distance=(
                float(nearest["distance"]) if nearest is not None else None
            ),
            target_contact=bool(target_contacts),
            any_key_contact=bool(any_key_contacts),
            active_action_names=active_names,
        )
        logs.append(
            ScriptedStepLog(
                step=step,
                target_keys=target_keys,
                selected_key=selected_key,
                selected_note=(
                    env.note_name_for_key(selected_key) if selected_key is not None else None
                ),
                selected_finger=selected_finger,
                pressed_keys=pressed_keys,
                target_key_state=target_state,
                max_unintended_key_state=max_unintended_state,
                nearest_fingertip=(
                    str(nearest["fingertip"]) if nearest is not None else None
                ),
                nearest_fingertip_distance=(
                    float(nearest["distance"]) if nearest is not None else None
                ),
                target_contact=bool(target_contacts),
                any_key_contact=bool(any_key_contacts),
                wrong_key_nearest_distance=wrong_key_distance,
                contact_pairs=tuple(_format_contact_pair(pair) for pair in any_key_contacts),
                reward=env.current_reward(),
                discount=timestep.discount,
                last=timestep.last(),
                status=diagnostic_category,
                diagnostic_category=diagnostic_category,
                active_action_names=active_names,
            )
        )
        if timestep.last():
            break

    return logs


def _classify_step(
    *,
    selected_key: int | None,
    pressed_keys: tuple[int, ...],
    target_key_state: float | None,
    max_unintended_key_state: float,
    nearest_fingertip_distance: float | None,
    target_contact: bool,
    any_key_contact: bool,
    active_action_names: tuple[str, ...],
) -> str:
    if selected_key is None:
        return "no_target"
    if selected_key in pressed_keys and len(pressed_keys) == 1:
        return "clean_target_press"
    if selected_key in pressed_keys:
        return "target_press_with_unintended_keys"
    if pressed_keys:
        return "unintended_wrong_key_region"
    if (
        max_unintended_key_state > _KEY_TRAVEL_EPSILON
        and target_key_state is not None
        and max_unintended_key_state > target_key_state + _KEY_TRAVEL_EPSILON
    ):
        return "unintended_wrong_key_region"
    if target_contact or (
        target_key_state is not None and target_key_state > _KEY_TRAVEL_EPSILON
    ):
        return "contact_occurred_but_key_travel_was_insufficient"
    if nearest_fingertip_distance is not None:
        if nearest_fingertip_distance > _APPROACH_DISTANCE_METERS:
            return "fingertip_never_approached_target_key"
        if not any_key_contact:
            return "fingertip_approached_but_no_contact"
    if not active_action_names:
        return "action_calibration_issue"
    if any_key_contact:
        return "unintended_wrong_key_region"
    return "other_unknown"


def _nearest_wrong_key_distance(
    env: ALAOneHandEnv,
    selected_key: int | None,
    pressed_keys: tuple[int, ...],
) -> float | None:
    wrong_keys = [key for key in pressed_keys if key != selected_key]
    distances = []
    for key in wrong_keys:
        nearest = env.nearest_fingertip_to_key(key)
        if nearest is not None:
            distances.append(float(nearest["distance"]))
    return min(distances) if distances else None


def _format_contact_pair(pair: tuple[str, str]) -> str:
    return f"{pair[0]} <-> {pair[1]}"
