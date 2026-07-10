from pathlib import Path

from ala_pianist.controllers.action_library import KEYSET_MIDI, build_action_library


ROOT = Path("/home/reece_dev/msc-audio-pianist")
LIBRARY_PATH = ROOT / "experiments" / "pipeline1" / "keyset_action_library.json"


def main() -> None:
    library = build_action_library(LIBRARY_PATH)
    print(f"library_path={LIBRARY_PATH}")
    print(f"keyset={list(KEYSET_MIDI)}")
    for pitch, entry in library.items():
        print(
            f"midi={pitch} key={entry.key_index} outcome={entry.outcome} "
            f"target={entry.max_target_key_state:.6f} "
            f"unintended={entry.max_unintended_key_state:.6f} "
            f"pressed={list(entry.pressed_keys)}"
        )


if __name__ == "__main__":
    main()
