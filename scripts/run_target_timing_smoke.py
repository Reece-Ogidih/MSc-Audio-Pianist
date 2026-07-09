from pathlib import Path

import numpy as np

from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


ROOT = Path("/home/reece_dev/msc-audio-pianist")
MIDI_PATH = ROOT / "tmp" / "target_timing_smoke.mid"


def key_index(pitch: int) -> int:
    return pitch - 21


def main() -> None:
    events = [
        NoteEvent(69, 0.00, 0.20, 80),
        NoteEvent(73, 0.25, 0.20, 80),
        NoteEvent(74, 0.50, 0.20, 80),
        NoteEvent(75, 0.75, 0.20, 80),
    ]
    write_monophonic_midi(events, MIDI_PATH, title="target timing smoke")

    env = ALAOneHandEnv(MIDI_PATH)
    timestep = env.reset()
    spec = env.action_spec()
    zero = np.zeros(spec.shape, dtype=spec.dtype)

    print(f"midi_path={MIDI_PATH}")
    print("expected_note_events:")
    for event in events:
        print(
            f"  pitch={event.pitch} key={key_index(event.pitch)} "
            f"start={event.start:.2f} duration={event.duration:.2f}"
        )
    print(f"action_shape={spec.shape}")
    print("observed_target_timeline:")

    for step in range(24):
        print(
            f"  step={step:02d} t_idx={env.task._t_idx:02d} "
            f"target_keys={env.current_target_keys()} "
            f"pressed_keys={env.current_pressed_keys()} "
            f"reward={env.current_reward()} "
            f"last={timestep.last()}"
        )
        timestep = env.step(zero)


if __name__ == "__main__":
    main()
