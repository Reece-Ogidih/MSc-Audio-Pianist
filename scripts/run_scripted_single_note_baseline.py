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
        selected = (
            "None"
            if log.selected_key is None
            else f"{log.selected_key} {log.selected_note}"
        )
        distance = (
            "None"
            if log.nearest_fingertip_distance is None
            else f"{log.nearest_fingertip_distance:.6f}"
        )
        target_state = (
            "None" if log.target_key_state is None else f"{log.target_key_state:.6f}"
        )
        print(
            f"  step={log.step:02d} target={list(log.target_keys)} "
            f"selected={selected} finger={log.selected_finger} "
            f"active={list(log.active_action_names)} "
            f"target_key_state={target_state} "
            f"max_unintended_key_state={log.max_unintended_key_state:.6f} "
            f"nearest_fingertip={log.nearest_fingertip} "
            f"nearest_target_distance_m={distance} "
            f"target_contact={log.target_contact} "
            f"any_key_contact={log.any_key_contact} "
            f"wrong_key_distance_m={log.wrong_key_nearest_distance} "
            f"pressed={list(log.pressed_keys)} reward={log.reward} "
            f"discount={log.discount} last={log.last} "
            f"category={log.diagnostic_category}"
        )
        if log.contact_pairs:
            print(f"    contact_pairs={list(log.contact_pairs[:4])}")

    counts = Counter(log.diagnostic_category for log in logs)
    print("summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"sustain_state={env.task.piano.sustain_state[0]}")


if __name__ == "__main__":
    main()
