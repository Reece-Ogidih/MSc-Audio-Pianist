from pathlib import Path

import numpy as np
from note_seq import midi_io, music_pb2

from ala_pianist.envs import OneHandRoboPianistEnv


ROOT = Path("/home/reece_dev/msc-audio-pianist")
MIDI_PATH = ROOT / "tmp" / "one_hand_env_smoke.mid"


def create_tiny_midi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = "one hand env smoke"
    seq.sequence_metadata.artist = "ALA Pianist"
    seq.tempos.add(qpm=90)
    for pitch, start, end in [
        (69, 0.00, 0.35),
        (73, 0.40, 0.75),
        (74, 0.80, 1.15),
        (75, 1.20, 1.55),
    ]:
        seq.notes.add(pitch=pitch, start_time=start, end_time=end, velocity=80, part=0)
    seq.total_time = 1.60
    midi_io.note_sequence_to_midi_file(seq, str(path))


def main() -> None:
    create_tiny_midi(MIDI_PATH)
    env = OneHandRoboPianistEnv(MIDI_PATH)
    timestep = env.reset()
    spec = env.action_spec()

    print(f"action_spec.shape={spec.shape}")
    print(f"action_spec.minimum={np.asarray(spec.minimum).tolist()}")
    print(f"action_spec.maximum={np.asarray(spec.maximum).tolist()}")
    print(f"observation_keys={sorted(timestep.observation.keys())}")

    zero_action = np.zeros(spec.shape, dtype=spec.dtype)
    for step in range(5):
        timestep = env.step(zero_action)
        print(
            f"step={step} target_keys={env.current_target_keys()} "
            f"pressed_keys={env.current_pressed_keys()} reward={env.current_reward()} "
            f"first={timestep.first()} mid={timestep.mid()} last={timestep.last()}"
        )


if __name__ == "__main__":
    main()
