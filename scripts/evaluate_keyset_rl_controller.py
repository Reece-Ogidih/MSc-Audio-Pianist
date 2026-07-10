import json
from pathlib import Path

import numpy as np

from ala_pianist.audio import synthesize_monophonic_wav, transcribe_monophonic_wav, transcription_accuracy
from ala_pianist.controllers.action_library import KEYSET_MIDI, build_action_library, load_action_library
from ala_pianist.controllers.rl_policy_controller import RLPolicyController
from ala_pianist.envs import ALAOneHandEnv
from ala_pianist.music import NoteEvent, write_monophonic_midi


ROOT = Path("/home/reece_dev/msc-audio-pianist")
OUT_DIR = ROOT / "experiments" / "keyset_rl"
PIPELINE_DIR = ROOT / "experiments" / "pipeline1"
MODEL_PATH = OUT_DIR / "keyset_sac_model.zip"
LIBRARY_PATH = PIPELINE_DIR / "keyset_action_library.json"
SUMMARY_PATH = OUT_DIR / "keyset_rl_evaluation_summary.json"


def default_events():
    return [
        NoteEvent(69, 0.00, 0.28, 90),
        NoteEvent(73, 0.40, 0.28, 90),
        NoteEvent(75, 0.80, 0.28, 90),
        NoteEvent(71, 1.20, 0.28, 90),
    ]


def evaluate_zero_per_note(horizon_steps: int = 32):
    return {
        pitch: _evaluate_fixed_action(pitch, None, horizon_steps=horizon_steps)
        for pitch in KEYSET_MIDI
    }


def evaluate_library_per_note(horizon_steps: int = 32):
    library = build_action_library(LIBRARY_PATH, midi_pitches=KEYSET_MIDI) if not LIBRARY_PATH.exists() else load_action_library(LIBRARY_PATH)
    return {
        pitch: _evaluate_fixed_action(pitch, np.asarray(library[pitch].action), horizon_steps=horizon_steps)
        for pitch in KEYSET_MIDI
    }


def evaluate_rl_per_note(model_path: Path, horizon_steps: int = 32):
    controller = RLPolicyController(model_path, deterministic=True, horizon_steps=horizon_steps)
    return {
        pitch: _evaluate_controller(pitch, controller, horizon_steps=horizon_steps)
        for pitch in KEYSET_MIDI
    }


def evaluate_full_pipeline_rl(model_path: Path, horizon_steps: int = 32):
    events = default_events()
    audio_path = OUT_DIR / "keyset_rl_pipeline_audio.wav"
    rollout_midi_path = OUT_DIR / "keyset_rl_pipeline_rollout.mid"
    synthesize_monophonic_wav(events, audio_path)
    transcription = transcribe_monophonic_wav(audio_path)
    accuracy = transcription_accuracy(events, transcription.note_events)
    transcribed = [event for event in transcription.note_events if event.pitch in KEYSET_MIDI]
    write_monophonic_midi(transcribed, rollout_midi_path, title="keyset RL pipeline evaluation")
    env = ALAOneHandEnv(rollout_midi_path)
    env.reset()
    controller = RLPolicyController(model_path, deterministic=True, horizon_steps=horizon_steps)
    note_metrics = []
    for event in transcribed:
        note_metrics.append(_rollout_controller_note(env, event.pitch, controller, horizon_steps))
    target_recall = sum(m["max_target_key_state"] >= 0.25 or m["key_index"] in m["pressed_keys"] for m in note_metrics) / max(1, len(note_metrics))
    wrong_key_rate = sum(bool([key for key in m["pressed_keys"] if key != m["key_index"]]) for m in note_metrics) / max(1, len(note_metrics))
    return {
        "transcription": accuracy,
        "expected_pitches": [event.pitch for event in events],
        "transcribed_pitches": [event.pitch for event in transcription.note_events],
        "note_metrics": note_metrics,
        "target_recall": target_recall,
        "wrong_key_rate": wrong_key_rate,
    }


def _evaluate_fixed_action(pitch: int, action, *, horizon_steps: int):
    midi_path = OUT_DIR / f"eval_note_{pitch}.mid"
    write_monophonic_midi([NoteEvent(pitch, 0.0, 1.0, 90)], midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    if action is None:
        action = np.zeros(env.action_spec().shape, dtype=env.action_spec().dtype)
    return _rollout_action(env, pitch, np.asarray(action, dtype=env.action_spec().dtype), horizon_steps)


def _evaluate_controller(pitch: int, controller: RLPolicyController, *, horizon_steps: int):
    midi_path = OUT_DIR / f"eval_rl_note_{pitch}.mid"
    write_monophonic_midi([NoteEvent(pitch, 0.0, 1.0, 90)], midi_path)
    env = ALAOneHandEnv(midi_path)
    env.reset()
    return _rollout_controller_note(env, pitch, controller, horizon_steps)


def _rollout_controller_note(env: ALAOneHandEnv, pitch: int, controller: RLPolicyController, horizon_steps: int):
    max_target = 0.0
    max_unintended = 0.0
    pressed = set()
    shaped_return = 0.0
    native_reward_sum = 0.0
    key = pitch - 21
    for step in range(horizon_steps):
        action = controller.action(env, target_midi=pitch, step_count=step)
        timestep = env.step(action)
        target = env.target_key_state(key) or 0.0
        unintended = env.max_unintended_key_state(key)
        pressed.update(env.current_pressed_keys())
        shaped_return += 8.0 * target - 4.0 * unintended - 2.0 * len([k for k in pressed if k != key])
        if env.current_reward() is not None:
            native_reward_sum += float(env.current_reward())
        max_target = max(max_target, target)
        max_unintended = max(max_unintended, unintended)
        if timestep.last():
            break
    return _metric(pitch, max_target, max_unintended, pressed, native_reward_sum, shaped_return)


def _rollout_action(env: ALAOneHandEnv, pitch: int, action: np.ndarray, horizon_steps: int):
    max_target = 0.0
    max_unintended = 0.0
    pressed = set()
    shaped_return = 0.0
    native_reward_sum = 0.0
    key = pitch - 21
    for step in range(horizon_steps):
        ramp = min(1.0, (step + 1) / 6.0)
        timestep = env.step(action * ramp)
        target = env.target_key_state(key) or 0.0
        unintended = env.max_unintended_key_state(key)
        pressed.update(env.current_pressed_keys())
        shaped_return += 8.0 * target - 4.0 * unintended - 2.0 * len([k for k in pressed if k != key])
        if env.current_reward() is not None:
            native_reward_sum += float(env.current_reward())
        max_target = max(max_target, target)
        max_unintended = max(max_unintended, unintended)
        if timestep.last():
            break
    return _metric(pitch, max_target, max_unintended, pressed, native_reward_sum, shaped_return)


def _metric(pitch: int, max_target: float, max_unintended: float, pressed: set[int], native_reward_sum: float, shaped_return: float):
    key = pitch - 21
    clean = pressed == {key}
    near = max_target >= 0.25 and max_unintended <= max_target + 0.02
    dirty = key in pressed and not clean
    missed = not clean and not dirty and not near
    return {
        "midi_pitch": pitch,
        "key_index": key,
        "max_target_key_state": float(max_target),
        "max_unintended_key_state": float(max_unintended),
        "pressed_keys": sorted(pressed),
        "clean_press": clean,
        "near_clean_partial_press": near,
        "dirty_press": dirty,
        "missed": missed,
        "native_reward_sum": float(native_reward_sum),
        "shaped_return": float(shaped_return),
        "outcome": "clean" if clean else "dirty" if dirty else "near_clean" if near else "missed",
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Expected trained model at {MODEL_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_path": str(MODEL_PATH),
        "keyset": list(KEYSET_MIDI),
        "zero_baseline": evaluate_zero_per_note(),
        "action_library": evaluate_library_per_note(),
        "rl_policy": evaluate_rl_per_note(MODEL_PATH),
        "pipeline1_rl": evaluate_full_pipeline_rl(MODEL_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"summary_path={SUMMARY_PATH}")
    print(f"model_path={MODEL_PATH}")
    print(f"keyset={list(KEYSET_MIDI)}")
    for label in ("zero_baseline", "action_library", "rl_policy"):
        print(f"{label}:")
        for pitch, metric in summary[label].items():
            print(
                f"  midi={pitch} outcome={metric['outcome']} "
                f"target={metric['max_target_key_state']:.6f} "
                f"unintended={metric['max_unintended_key_state']:.6f} "
                f"pressed={metric['pressed_keys']}"
            )
    pipe = summary["pipeline1_rl"]
    print(f"pipeline_transcription={pipe['transcription']}")
    print(f"pipeline_target_recall={pipe['target_recall']:.6f}")
    print(f"pipeline_wrong_key_rate={pipe['wrong_key_rate']:.6f}")
    print(f"pipeline_note_metrics={pipe['note_metrics']}")


if __name__ == "__main__":
    main()
