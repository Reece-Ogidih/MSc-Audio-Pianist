from pathlib import Path
import importlib.util

import pytest

from ala_pianist.audio.transcriber import OracleMidiTranscriber
from ala_pianist.audio.transcriber import basic_pitch_notes_to_timed_notes
from ala_pianist.evaluation.transcription_metrics import transcription_note_metrics
from ala_pianist.music import NoteEvent, timing_quantization_report, write_monophonic_midi
from ala_pianist.music.timed_notes import (
    MonophonicCleanupConfig,
    TimedNote,
    clean_monophonic_timed_notes,
    normalize_timed_notes,
    note_event_to_timed_note,
    timed_notes_to_controller_sequence,
    write_controller_goal_midi,
)
from ala_pianist.pipelines.indirect import (
    BENCHMARK_SEQUENCE_PITCHES,
    benchmark_sequence_events,
    create_rendered_benchmark,
    note_events_to_timed_notes,
    run_symbolic_frontend,
    write_benchmark_sequence_midi,
)


def test_timed_note_conversion_preserves_timing() -> None:
    note = TimedNote(73, 0.123, 0.456, confidence=0.8, source="test")
    sequence = timed_notes_to_controller_sequence([note])

    assert sequence.pitches == (73,)
    assert sequence.key_indices == (52,)
    assert sequence.note_events[0].start == pytest.approx(0.123)
    assert sequence.note_events[0].duration == pytest.approx(0.333)
    assert sequence.timing_preserved is True


def test_range_handling_drop_clip_and_error() -> None:
    notes = [TimedNote(71, 0.0, 0.2), TimedNote(73, 0.3, 0.5)]

    with pytest.raises(ValueError, match="outside supported range"):
        normalize_timed_notes(notes, midi_min=72, midi_max=76)

    dropped = normalize_timed_notes(notes, midi_min=72, midi_max=76, range_policy="drop")
    assert [note.pitch for note in dropped] == [73]

    clipped = normalize_timed_notes(notes, midi_min=72, midi_max=76, range_policy="clip")
    assert [note.pitch for note in clipped] == [72, 73]
    assert clipped[0].metadata["original_pitch"] == 71


def test_confidence_threshold_and_empty_predictions() -> None:
    notes = [
        TimedNote(73, 0.0, 0.2, confidence=0.4),
        TimedNote(74, 0.3, 0.5, confidence=0.9),
    ]
    filtered = normalize_timed_notes(notes, confidence_threshold=0.5)
    assert [note.pitch for note in filtered] == [74]

    metrics = transcription_note_metrics(expected=[notes[1]], predicted=[])
    assert metrics.note_precision == 0.0
    assert metrics.note_recall == 0.0
    assert metrics.note_f1 == 0.0


def test_duplicate_notes_can_merge_or_error() -> None:
    notes = [
        TimedNote(73, 0.0, 0.2, confidence=0.5),
        TimedNote(73, 0.0, 0.2, confidence=0.9),
    ]
    merged = normalize_timed_notes(notes, duplicate_policy="merge")
    assert len(merged) == 1
    assert merged[0].confidence == pytest.approx(0.9)
    assert merged[0].metadata["duplicate_count"] == 2

    with pytest.raises(ValueError, match="Duplicate timed note"):
        normalize_timed_notes(notes, duplicate_policy="error")


def test_polyphonic_predictions_are_preserved_for_downstream_error_accounting(tmp_path: Path) -> None:
    notes = [
        TimedNote(73, 0.0, 0.3, confidence=0.9),
        TimedNote(74, 0.1, 0.2, confidence=0.8),
    ]
    sequence = timed_notes_to_controller_sequence(notes, allow_polyphony=True)
    midi_path = write_controller_goal_midi(sequence, tmp_path / "polyphonic_prediction.mid")

    assert sequence.monophonic is False
    assert midi_path.exists()

    with pytest.raises(ValueError, match="monophonic"):
        normalize_timed_notes(notes, allow_polyphony=False)


def test_transcription_metrics_extra_and_missing_notes() -> None:
    expected = [TimedNote(73, 0.0, 0.2), TimedNote(75, 0.4, 0.6)]
    extra = expected + [TimedNote(76, 0.8, 1.0)]
    extra_metrics = transcription_note_metrics(expected, extra)
    assert extra_metrics.note_recall == 1.0
    assert extra_metrics.note_precision == pytest.approx(2 / 3)
    assert extra_metrics.note_f1 < 1.0

    missing_metrics = transcription_note_metrics(expected, [expected[0]])
    assert missing_metrics.note_precision == 1.0
    assert missing_metrics.note_recall == pytest.approx(0.5)
    assert missing_metrics.note_f1 < 1.0
    assert extra_metrics.false_positive_pitches == (76,)
    assert missing_metrics.false_negative_pitches == (75,)
    assert extra_metrics.per_pitch[76]["precision"] == 0.0


def test_basic_pitch_raw_note_conversion_filters_invalid_durations() -> None:
    notes, invalid = basic_pitch_notes_to_timed_notes(
        [
            (0.0, 0.25, 73, 0.8, None),
            (0.3, 0.3, 74, 0.7, None),
            (0.4, 0.35, 75, 0.7, None),
        ]
    )

    assert invalid == 2
    assert len(notes) == 1
    assert notes[0].pitch == 73
    assert notes[0].confidence == pytest.approx(0.8)


def test_oracle_midi_adapter_uses_same_downstream_interface(tmp_path: Path) -> None:
    midi_path = tmp_path / "oracle.mid"
    events = (NoteEvent(73, 0.0, 0.28, 90), NoteEvent(75, 0.4, 0.28, 90))
    write_monophonic_midi(events, midi_path)

    result = run_symbolic_frontend(
        audio_path=tmp_path / "unused.wav",
        transcriber=OracleMidiTranscriber(midi_path),
        expected_notes=tuple(note_event_to_timed_note(event) for event in events),
    )

    assert result.controller_sequence.pitches == (73, 75)
    assert result.controller_sequence.key_indices == (52, 54)
    assert result.transcription_metrics is not None
    assert result.transcription_metrics.note_f1 == 1.0
    assert result.controller_sequence.note_events[1].start == pytest.approx(0.4, abs=1e-5)


def test_oracle_and_predicted_paths_converge_on_same_controller_representation(tmp_path: Path) -> None:
    midi_path = tmp_path / "oracle.mid"
    events = (NoteEvent(73, 0.0, 0.28, 90), NoteEvent(75, 0.4, 0.28, 90))
    write_monophonic_midi(events, midi_path)

    oracle = run_symbolic_frontend(
        audio_path=tmp_path / "unused.wav",
        transcriber=OracleMidiTranscriber(midi_path),
    )
    predicted = timed_notes_to_controller_sequence(
        tuple(note_event_to_timed_note(event, source="mock_transcriber") for event in events)
    )

    assert oracle.controller_sequence.pitches == predicted.pitches
    assert oracle.controller_sequence.key_indices == predicted.key_indices
    assert [event.start for event in oracle.controller_sequence.note_events] == pytest.approx(
        [event.start for event in predicted.note_events],
        abs=1e-5,
    )


def test_benchmark_sequences_are_inside_initial_indirect_range(tmp_path: Path) -> None:
    for events in [
        benchmark_sequence_events((72,)),
        benchmark_sequence_events((73, 74)),
        benchmark_sequence_events((75, 76)),
    ]:
        notes = note_events_to_timed_notes(events)
        assert all(72 <= note.pitch <= 76 for note in notes)

    midi_path = write_benchmark_sequence_midi((73, 74, 75), tmp_path / "benchmark.mid")
    assert midi_path.exists()


def test_benchmark_contains_all_five_pitches_and_all_eight_transitions() -> None:
    assert {pitch for sequence in BENCHMARK_SEQUENCE_PITCHES for pitch in sequence} == {72, 73, 74, 75, 76}
    transitions = {sequence for sequence in BENCHMARK_SEQUENCE_PITCHES if len(sequence) == 2}
    assert transitions == {
        (72, 73),
        (73, 72),
        (73, 74),
        (74, 73),
        (74, 75),
        (75, 74),
        (75, 76),
        (76, 75),
    }


def test_manifest_generation_uses_no_sustain_and_mocked_renderer(tmp_path: Path, monkeypatch) -> None:
    def fake_render(midi_path, wav_path, *, soundfont_path, sample_rate, gain):
        path = Path(wav_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake wav")
        return path

    monkeypatch.setattr("ala_pianist.pipelines.indirect.render_midi_with_fluidsynth", fake_render)
    items = create_rendered_benchmark(
        tmp_path,
        soundfont_path=tmp_path / "fake.sf2",
        sample_rate=16000,
    )

    manifest = (tmp_path / "benchmark_manifest.json").read_text()
    assert len(items) == 13
    assert '"sustain": "none"' in manifest
    assert '"midi_min": 72' in manifest
    assert '"midi_max": 76' in manifest


def test_timing_quantization_is_explicit() -> None:
    sequence = timed_notes_to_controller_sequence([TimedNote(73, 0.013, 0.281)])
    report = timing_quantization_report(sequence, control_timestep_seconds=0.01)

    assert report["event_count"] == 2
    assert report["max_abs_error_seconds"] > 0.0
    assert report["control_timestep_seconds"] == 0.01


def test_monophonic_cleanup_suppresses_repeated_same_pitch_predictions() -> None:
    raw = [
        TimedNote(73, 0.01, 0.60, confidence=0.7),
        TimedNote(73, 1.50, 2.00, confidence=0.9),
    ]
    cleaned, diagnostics = clean_monophonic_timed_notes(raw)

    assert [note.pitch for note in cleaned] == [73]
    assert cleaned[0].onset == pytest.approx(0.01)
    assert cleaned[0].confidence == pytest.approx(0.9)
    assert diagnostics.same_pitch_duplicate_suppressed_count == 1
    assert diagnostics.cleaned_same_pitch_duplicate_count == 0


def test_monophonic_cleanup_truncates_previous_offset_at_next_onset() -> None:
    raw = [
        TimedNote(73, 0.01, 0.70),
        TimedNote(74, 0.40, 1.00),
    ]
    cleaned, diagnostics = clean_monophonic_timed_notes(raw)

    assert [note.pitch for note in cleaned] == [73, 74]
    assert [note.onset for note in cleaned] == pytest.approx([0.01, 0.40])
    assert cleaned[0].offset == pytest.approx(0.40)
    assert cleaned[1].offset == pytest.approx(1.00)
    assert diagnostics.overlap_truncation_count == 1
    assert diagnostics.cleaned_overlap_count == 0


def test_monophonic_cleanup_removes_invalid_and_cannot_create_notes() -> None:
    raw = [
        TimedNote(72, 0.00, 0.25),
    ]
    invalid_like = [
        TimedNote(72, 0.00, 0.25),
        TimedNote(73, 0.30, 0.31),
    ]
    cleaned, diagnostics = clean_monophonic_timed_notes(raw)
    clipped, clipped_diagnostics = clean_monophonic_timed_notes(
        invalid_like,
        config=MonophonicCleanupConfig(audio_duration_seconds=0.30),
    )

    assert len(cleaned) <= len(raw)
    assert {note.pitch for note in cleaned}.issubset({note.pitch for note in raw})
    assert [note.pitch for note in clipped] == [72]
    assert clipped_diagnostics.invalid_duration_count == 1


def test_monophonic_cleanup_disabled_path_is_identity() -> None:
    raw = [
        TimedNote(73, 0.01, 0.70),
        TimedNote(74, 0.40, 1.00),
    ]
    cleaned, diagnostics = clean_monophonic_timed_notes(
        raw,
        config=MonophonicCleanupConfig(
            suppress_repeated_same_pitch=False,
            truncate_overlaps=False,
        ),
    )

    assert cleaned == tuple(raw)
    assert diagnostics.overlap_truncation_count == 0
    assert diagnostics.same_pitch_duplicate_suppressed_count == 0


def test_benchmark_runner_reports_transcriber_failure(tmp_path: Path) -> None:
    script_path = Path("/home/reece_dev/msc-audio-pianist/scripts/run_indirect_five_note_benchmark.py")
    spec = importlib.util.spec_from_file_location("run_indirect_five_note_benchmark", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class FailingTranscriber:
        name = "failing"

        def transcribe(self, audio_path):
            raise RuntimeError("boom")

    output = module._safe_transcribe(FailingTranscriber(), tmp_path / "audio.wav")

    assert output.notes == ()
    assert output.transcriber_name == "failing"
    assert "RuntimeError: boom" in output.metadata["error"]
