from ala_pianist.controllers.action_library import build_action_library, load_action_library


def test_action_library_builds_and_roundtrips(tmp_path):
    path = tmp_path / "library.json"
    library = build_action_library(
        path,
        midi_pitches=(75,),
        candidate_count=2,
        horizon_steps=4,
        seed=3,
    )
    loaded = load_action_library(path)

    assert path.exists()
    assert 75 in library
    assert 75 in loaded
    assert loaded[75].key_index == 75 - 21
    assert len(loaded[75].action) == 22
    assert loaded[75].outcome
