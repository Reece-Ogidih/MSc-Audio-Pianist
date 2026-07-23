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

__all__ = [
    "CURRICULUM_MODES",
    "CurriculumClip",
    "CurriculumPhase",
    "NoteEvent",
    "SequenceTimingConfig",
    "assign_right_hand_fingering",
    "generate_curriculum_events",
    "generate_sequence_events",
    "load_staged_curriculum",
    "note_windows",
    "phase_for_global_step",
    "phases_as_dicts",
    "sequence_timing_from_profile",
    "write_curriculum_midi",
    "write_monophonic_midi",
    "write_sequence_midi",
]
