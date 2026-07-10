from pathlib import Path

from ala_pianist.controllers.action_library import KEYSET_MIDI, load_action_library
from ala_pianist.pipelines import run_pipeline1


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "pipeline1"
AUDIO_PATH = OUT_DIR / "pipeline1_generated.wav"
LIBRARY_PATH = OUT_DIR / "keyset_action_library.json"
ROLLOUT_MIDI_PATH = OUT_DIR / "pipeline1_transcribed.mid"
SUMMARY_PATH = OUT_DIR / "pipeline1_summary.json"


def main() -> None:
    result = run_pipeline1(
        audio_path=AUDIO_PATH,
        library_path=LIBRARY_PATH,
        rollout_midi_path=ROLLOUT_MIDI_PATH,
        summary_path=SUMMARY_PATH,
        build_library=True,
    )
    library = load_action_library(LIBRARY_PATH)

    print(f"audio_path={AUDIO_PATH}")
    print(f"library_path={LIBRARY_PATH}")
    print(f"summary_path={SUMMARY_PATH}")
    print(f"keyset={list(KEYSET_MIDI)}")
    print(f"expected_pitches={list(result.expected_pitches)}")
    print(f"transcribed_pitches={list(result.transcribed_pitches)}")
    print(f"transcription={result.transcription}")
    print("action_library:")
    for pitch in KEYSET_MIDI:
        entry = library[pitch]
        print(
            f"  midi={pitch} key={entry.key_index} outcome={entry.outcome} "
            f"target={entry.max_target_key_state:.6f} "
            f"unintended={entry.max_unintended_key_state:.6f} "
            f"pressed={list(entry.pressed_keys)}"
        )
    print("rollout_metrics:")
    for metric in result.note_metrics:
        print(
            f"  midi={metric.midi_pitch} key={metric.key_index} "
            f"outcome={metric.outcome} "
            f"target={metric.max_target_key_state:.6f} "
            f"unintended={metric.max_unintended_key_state:.6f} "
            f"pressed={list(metric.pressed_keys)}"
        )
    print(f"target_recall={result.target_recall:.6f}")
    print(f"wrong_key_rate={result.wrong_key_rate:.6f}")


if __name__ == "__main__":
    main()
