from ala_pianist.audio import synthesize_monophonic_wav, transcribe_monophonic_wav, transcription_accuracy
from ala_pianist.music import NoteEvent


def test_generated_audio_transcribes_monophonic_notes(tmp_path):
    events = [
        NoteEvent(69, 0.0, 0.25, 90),
        NoteEvent(73, 0.4, 0.25, 90),
        NoteEvent(75, 0.8, 0.25, 90),
    ]
    wav_path = tmp_path / "notes.wav"
    synthesize_monophonic_wav(events, wav_path)
    result = transcribe_monophonic_wav(wav_path)
    accuracy = transcription_accuracy(events, result.note_events)

    assert wav_path.exists()
    assert accuracy["observed_pitches"][:3] == [69, 73, 75]
    assert accuracy["pitch_accuracy"] == 1.0
