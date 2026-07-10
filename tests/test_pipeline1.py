from ala_pianist.music import NoteEvent
from ala_pianist.pipelines import run_pipeline1


def test_pipeline1_runs_end_to_end(tmp_path):
    result = run_pipeline1(
        audio_path=tmp_path / "clip.wav",
        library_path=tmp_path / "library.json",
        rollout_midi_path=tmp_path / "rollout.mid",
        summary_path=tmp_path / "summary.json",
        note_events=[
            NoteEvent(69, 0.0, 0.25, 90),
            NoteEvent(73, 0.4, 0.25, 90),
        ],
        build_library=True,
    )

    assert result.transcription["pitch_accuracy"] == 1.0
    assert result.note_metrics
    assert 0.0 <= result.target_recall <= 1.0
    assert 0.0 <= result.wrong_key_rate <= 1.0
