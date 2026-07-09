from collections import Counter
from pathlib import Path

from ala_pianist.baselines import ScriptedDiagnosticController, run_scripted_diagnostic
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


ROOT = Path("/home/reece_dev/msc-audio-pianist")
MIDI_PATH = ROOT / "tmp" / "scripted_single_note_baseline.mid"


def main() -> None:
    events = [
        NoteEvent(69, 0.00, 0.20, 80),
        NoteEvent(73, 0.25, 0.20, 80),
        NoteEvent(74, 0.50, 0.20, 80),
        NoteEvent(75, 0.75, 0.20, 80),
    ]
    write_monophonic_midi(events, MIDI_PATH, title="scripted diagnostic baseline")

    env = ALAOneHandEnv(MIDI_PATH)
    spec = env.action_spec()
    names = env.action_names()
    controller = ScriptedDiagnosticController()
    logs = run_scripted_diagnostic(env, controller, max_steps=64)

    print(f"midi_path={MIDI_PATH}")
    print("expected_events:")
    for event in events:
        print(
            f"  pitch={event.pitch} key={event.pitch - 21} "
            f"start={event.start:.2f} duration={event.duration:.2f}"
        )

    print(f"action_shape={spec.shape}")
    print("action_dimensions:")
    for idx, name in enumerate(names):
        print(
            f"  {idx:02d} {name} "
            f"bounds=({spec.minimum[idx]:.6f}, {spec.maximum[idx]:.6f})"
        )

    print("rollout:")
    for log in logs:
        print(
            f"  step={log.step:02d} target={list(log.target_keys)} "
            f"selected={log.selected_key} finger={log.selected_finger} "
            f"pressed={list(log.pressed_keys)} reward={log.reward} "
            f"discount={log.discount} last={log.last} status={log.status} "
            f"active={list(log.active_action_names)}"
        )

    counts = Counter(log.status for log in logs)
    print("summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"sustain_state={env.task.piano.sustain_state[0]}")


if __name__ == "__main__":
    main()
