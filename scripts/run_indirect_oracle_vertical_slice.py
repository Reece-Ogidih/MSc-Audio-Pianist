"""Run a tiny indirect-pipeline frontend vertical slice on generated audio."""

from __future__ import annotations

from pathlib import Path

from ala_pianist.audio import synthesize_monophonic_wav
from ala_pianist.audio.transcriber import GeneratedWavPeakTranscriber, OracleMidiTranscriber
from ala_pianist.pipelines.indirect import (
    benchmark_sequence_events,
    note_events_to_timed_notes,
    run_symbolic_frontend,
    write_benchmark_sequence_midi,
    write_controller_midi_from_result,
    write_symbolic_result_summary,
)


def main() -> None:
    root = Path("experiments/indirect_pipeline/oracle_vertical_slice")
    root.mkdir(parents=True, exist_ok=True)
    pitches = (73, 74, 75)
    events = benchmark_sequence_events(pitches)
    midi_path = write_benchmark_sequence_midi(pitches, root / "ground_truth.mid")
    wav = synthesize_monophonic_wav(events, root / "rendered_audio.wav")
    expected = note_events_to_timed_notes(events)

    oracle = run_symbolic_frontend(
        audio_path=wav.path,
        transcriber=OracleMidiTranscriber(midi_path),
        expected_notes=expected,
    )
    oracle_midi = write_controller_midi_from_result(oracle, root / "oracle_controller_goal.mid")
    write_symbolic_result_summary(oracle, root / "oracle_summary.json", controller_midi_path=oracle_midi)

    predicted = run_symbolic_frontend(
        audio_path=wav.path,
        transcriber=GeneratedWavPeakTranscriber(),
        expected_notes=expected,
    )
    predicted_midi = write_controller_midi_from_result(
        predicted,
        root / "predicted_controller_goal.mid",
    )
    write_symbolic_result_summary(
        predicted,
        root / "predicted_summary.json",
        controller_midi_path=predicted_midi,
    )

    print("Indirect oracle vertical slice")
    print(f"pitches: {list(pitches)}")
    print(f"ground_truth_midi: {midi_path}")
    print(f"rendered_audio: {wav.path}")
    print(f"oracle_controller_goal: {oracle_midi}")
    print(f"predicted_controller_goal: {predicted_midi}")
    print(f"oracle_metrics: {oracle.transcription_metrics.as_dict()}")
    print(f"predicted_metrics: {predicted.transcription_metrics.as_dict()}")


if __name__ == "__main__":
    main()
