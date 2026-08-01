"""Music utilities for ALA Pianist."""

from ala_pianist.music.curriculum import (
    CURRICULUM_MODES,
    CurriculumClip,
    assign_right_hand_fingering,
    generate_curriculum_events,
    write_curriculum_midi,
)
from ala_pianist.music.midi_utils import NoteEvent, write_monophonic_midi
from ala_pianist.music.sequence_generation import (
    SequenceTimingConfig,
    generate_sequence_events,
    note_windows,
    sequence_timing_from_profile,
    write_sequence_midi,
)
from ala_pianist.music.staged_curriculum import (
    CurriculumPhase,
    load_staged_curriculum,
    phase_for_global_step,
    phases_as_dicts,
)
from ala_pianist.music.timed_notes import (
    ControllerGoalSequence,
    MonophonicCleanupConfig,
    MonophonicCleanupDiagnostics,
    TimedNote,
    clean_monophonic_timed_notes,
    count_overlapping_notes,
    count_repeated_same_pitch_notes,
    normalize_timed_notes,
    note_event_to_timed_note,
    timed_note_to_note_event,
    timed_notes_to_controller_sequence,
    timing_quantization_report,
    write_controller_goal_midi,
    write_timed_notes_midi,
)

__all__ = [
    "CURRICULUM_MODES",
    "ControllerGoalSequence",
    "CurriculumClip",
    "CurriculumPhase",
    "MonophonicCleanupConfig",
    "MonophonicCleanupDiagnostics",
    "NoteEvent",
    "SequenceTimingConfig",
    "TimedNote",
    "assign_right_hand_fingering",
    "clean_monophonic_timed_notes",
    "count_overlapping_notes",
    "count_repeated_same_pitch_notes",
    "generate_curriculum_events",
    "generate_sequence_events",
    "load_staged_curriculum",
    "normalize_timed_notes",
    "note_windows",
    "note_event_to_timed_note",
    "phase_for_global_step",
    "phases_as_dicts",
    "sequence_timing_from_profile",
    "timed_note_to_note_event",
    "timed_notes_to_controller_sequence",
    "timing_quantization_report",
    "write_controller_goal_midi",
    "write_curriculum_midi",
    "write_monophonic_midi",
    "write_sequence_midi",
    "write_timed_notes_midi",
]
