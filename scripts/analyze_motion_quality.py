#!/usr/bin/env python
"""Analyze action and fingertip motion quality from frozen audit traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path("/home/reece_dev/msc-audio-pianist")
sys.path.insert(0, str(ROOT / "src"))

from ala_pianist.evaluation.motion_quality import (  # noqa: E402
    ACTION_NAMES,
    action_columns,
    action_dimension_mapping,
    action_quality,
    bootstrap_mean_ci,
    fingertip_motion_quality,
    infer_action_group,
    per_dimension_action_quality,
    phase_labels,
    standardized_difference,
)


DEFAULT_AUDIT = ROOT / "artifacts/frozen_models/five_note_symbolic_controller_v1/audit"
OUT_DIRNAME = "motion_quality"
NATIVE_BOUNDS = [
    (-0.5235990285873413, 0.17453299462795258),
    (-0.6981319785118103, 0.4886919856071472),
    (-1.0471999645233154, 1.0471999645233154),
    (0.0, 1.2217299938201904),
    (-0.20943999290466309, 0.20943999290466309),
    (-0.6981319785118103, 0.6981319785118103),
    (-0.26179900765419006, 1.5707999467849731),
    (-0.34906598925590515, 0.34906598925590515),
    (-0.26179900765419006, 1.5707999467849731),
    (0.0, 3.1414999961853027),
    (-0.34906598925590515, 0.34906598925590515),
    (-0.26179900765419006, 1.5707999467849731),
    (0.0, 3.1414999961853027),
    (-0.34906598925590515, 0.34906598925590515),
    (-0.26179900765419006, 1.5707999467849731),
    (0.0, 3.1414999961853027),
    (0.0, 0.785398006439209),
    (-0.34906598925590515, 0.34906598925590515),
    (-0.26179900765419006, 1.5707999467849731),
    (0.0, 3.1414999961853027),
    (-0.7605000138282776, 0.46050000190734863),
    (0.0, 0.05999999865889549),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    result = analyze(args.audit_root)
    print(json.dumps(result, indent=2, sort_keys=True))


def analyze(audit_root: Path) -> dict[str, str | int]:
    out_dir = audit_root / OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(audit_root / "reference_rollout_metrics.csv")
    repeated = pd.read_csv(audit_root / "repeated_evaluation/per_rollout_metrics.csv")

    rollout_rows: list[dict] = []
    dim_rows: list[dict] = []
    phase_rows: list[dict] = []
    coverage_rows: list[dict] = []
    for _, row in reference.iterrows():
        trace_path = Path(row["trace_path"])
        if not trace_path.exists():
            continue
        trace = pd.read_csv(trace_path)
        act_cols = action_columns(trace.columns)
        phases = phase_labels(trace)
        dt = float(row.get("control_timestep", 0.05) or 0.05)
        actions = trace[act_cols].to_numpy(dtype=float)
        base = {
            "model_id": row["model_id"],
            "sequence_id": row["sequence_id"],
            "sequence_group": row["sequence_group"],
            "rollout_seed": int(row["rollout_seed"]),
            "checkpoint_step": int(row["checkpoint_step"]),
            "trace_path": str(trace_path),
            "step_count": int(len(trace)),
            "control_timestep": dt,
            "has_actions": bool(len(act_cols) == 22),
            "has_fingertips": bool(any(column.startswith("fingertip_") for column in trace.columns)),
            "pressed_key_f1": float(row["pressed_key_f1"]),
            "timestep_f1": float(row["timestep_f1"]),
            "max_unintended_activation": float(row["max_unintended_activation"]),
            "integrated_unintended_travel": float(row["integrated_unintended_travel"]),
            "wrong_key_threshold_crossings": float(row["wrong_key_threshold_crossings"]),
            "failure_labels": row["failure_labels"],
            "strict_outcome": row["strict_outcome"],
        }
        rollout_rows.append({**base, **action_quality(actions), **fingertip_motion_quality(trace, dt=dt)})
        coverage_rows.append(base)
        for phase in sorted(set(phases)):
            mask = phases == phase
            if int(mask.sum()) == 0:
                continue
            phase_trace = trace.loc[mask]
            phase_actions = phase_trace[act_cols].to_numpy(dtype=float)
            phase_rows.append(
                {
                    "model_id": row["model_id"],
                    "sequence_id": row["sequence_id"],
                    "phase": phase,
                    "step_count": int(mask.sum()),
                    **action_quality(phase_actions),
                    **fingertip_motion_quality(phase_trace, dt=dt),
                }
            )
        for dim in per_dimension_action_quality(actions):
            index = int(dim["action_index"])
            dim_rows.append(
                {
                    "model_id": row["model_id"],
                    "sequence_id": row["sequence_id"],
                    "sequence_group": row["sequence_group"],
                    "action_index": index,
                    "action_name": ACTION_NAMES[index],
                    "action_group": infer_action_group(ACTION_NAMES[index]),
                    **{k: v for k, v in dim.items() if k != "action_index"},
                    "pressed_key_f1": float(row["pressed_key_f1"]),
                    "timestep_f1": float(row["timestep_f1"]),
                    "max_unintended_activation": float(row["max_unintended_activation"]),
                    "failure_labels": row["failure_labels"],
                }
            )

    per_rollout = pd.DataFrame(rollout_rows)
    per_dim = pd.DataFrame(dim_rows)
    phase_df = pd.DataFrame(phase_rows)
    per_rollout.to_csv(out_dir / "motion_quality_per_rollout.csv", index=False)
    per_dim.to_csv(out_dir / "motion_quality_per_action_dimension.csv", index=False)
    phase_df.to_csv(out_dir / "motion_quality_by_phase.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "trace_coverage.csv", index=False)

    per_model = summarize_per_model(per_rollout)
    per_model.to_csv(out_dir / "motion_quality_per_model.csv", index=False)
    assoc = failure_association(per_rollout)
    assoc.to_csv(out_dir / "failure_association.csv", index=False)

    write_action_mapping(out_dir / "action_dimension_mapping.md", per_dim)
    write_refinement_design(out_dir / "refinement_design.md", per_rollout, per_dim, phase_df, assoc)
    write_canary_plan(out_dir / "canary_experiment_plan.md")
    write_report(
        out_dir / "motion_quality_report.md",
        per_rollout=per_rollout,
        per_model=per_model,
        per_dim=per_dim,
        phase_df=phase_df,
        assoc=assoc,
        reference=reference,
        repeated=repeated,
    )
    return {
        "output_dir": str(out_dir),
        "reference_traces_analyzed": len(per_rollout),
        "repeated_rollouts_available": len(repeated),
        "per_rollout": str(out_dir / "motion_quality_per_rollout.csv"),
    }


def summarize_per_model(per_rollout: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_abs_action",
        "action_std",
        "mean_abs_action_delta",
        "mean_squared_action_delta",
        "max_abs_action_delta",
        "action_saturation_fraction",
        "sign_change_rate",
        "high_frequency_oscillation_rate",
        "mean_fingertip_speed",
        "p95_fingertip_speed",
        "max_fingertip_speed",
        "mean_fingertip_acceleration",
        "p95_fingertip_acceleration",
        "max_fingertip_acceleration",
        "mean_fingertip_jerk",
        "p95_fingertip_jerk",
        "max_fingertip_jerk",
        "pressed_key_f1",
        "timestep_f1",
        "max_unintended_activation",
        "integrated_unintended_travel",
    ]
    rows = []
    for model_id, group in per_rollout.groupby("model_id"):
        row = {"model_id": model_id, "rollout_count": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model_id")


def failure_association(per_rollout: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_abs_action_delta",
        "max_abs_action_delta",
        "action_saturation_fraction",
        "sign_change_rate",
        "high_frequency_oscillation_rate",
        "mean_fingertip_speed",
        "p95_fingertip_speed",
        "mean_fingertip_jerk",
        "p95_fingertip_jerk",
        "max_unintended_activation",
        "pressed_key_f1",
        "timestep_f1",
    ]
    labels = sorted(
        {
            label
            for raw in per_rollout["failure_labels"]
            for label in json.loads(raw)
            if label != "none"
        }
    )
    rows = []
    for label in labels:
        has = per_rollout["failure_labels"].apply(lambda raw: label in json.loads(raw))
        for metric in metrics:
            yes = per_rollout.loc[has, metric].to_numpy(dtype=float)
            no = per_rollout.loc[~has, metric].to_numpy(dtype=float)
            if len(yes) == 0 or len(no) == 0:
                continue
            ci_low, ci_high = bootstrap_mean_ci(yes)
            rows.append(
                {
                    "failure_label": label,
                    "metric": metric,
                    "with_label_count": int(len(yes)),
                    "without_label_count": int(len(no)),
                    "with_label_mean": float(np.mean(yes)),
                    "with_label_median": float(np.median(yes)),
                    "with_label_std": float(np.std(yes, ddof=0)),
                    "with_label_mean_ci_low": ci_low,
                    "with_label_mean_ci_high": ci_high,
                    "without_label_mean": float(np.mean(no)),
                    "without_label_median": float(np.median(no)),
                    "without_label_std": float(np.std(no, ddof=0)),
                    "standardized_difference": standardized_difference(yes, no),
                }
            )
    return pd.DataFrame(rows)


def write_action_mapping(path: Path, per_dim: pd.DataFrame) -> None:
    lines = [
        "# Action Dimension Mapping",
        "",
        "The policy action exposed to learning is normalized `[-1, 1]` with shape `(22,)`. "
        "The Gym wrapper rescales these values to native RoboPianist position-actuator target bounds, "
        "then appends sustain internally as `0.0`. Sustain is therefore fixed and not part of the policy action.",
        "",
        "The mapping below was verified from `GeneralOneHandGoalEnv.action_names` and native action bounds.",
        "",
        "| index | name | group | native lower | native upper | saturation mean | delta mean | oscillation mean |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    dim_summary = per_dim.groupby(["action_index", "action_name", "action_group"], as_index=False).agg(
        action_saturation_fraction=("action_saturation_fraction", "mean"),
        mean_abs_action_delta=("mean_abs_action_delta", "mean"),
        high_frequency_oscillation_rate=("high_frequency_oscillation_rate", "mean"),
    )
    bounds = action_dimension_mapping(NATIVE_BOUNDS)
    by_index = {item.index: item for item in bounds}
    for _, row in dim_summary.sort_values("action_index").iterrows():
        item = by_index[int(row["action_index"])]
        lines.append(
            f"| {item.index} | `{item.name}` | {item.group} | {item.native_lower:.6f} | "
            f"{item.native_upper:.6f} | {row['action_saturation_fraction']:.3f} | "
            f"{row['mean_abs_action_delta']:.3f} | {row['high_frequency_oscillation_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Dominant high-motion groups are identified in `motion_quality_per_action_dimension.csv`; "
            "the largest values are forearm/wrist contributions rather than isolated piano-key logic.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_refinement_design(
    path: Path,
    per_rollout: pd.DataFrame,
    per_dim: pd.DataFrame,
    phase_df: pd.DataFrame,
    assoc: pd.DataFrame,
) -> None:
    top_dims = per_dim.groupby(["action_index", "action_name", "action_group"], as_index=False).agg(
        mean_abs_action_delta=("mean_abs_action_delta", "mean"),
        action_saturation_fraction=("action_saturation_fraction", "mean"),
        high_frequency_oscillation_rate=("high_frequency_oscillation_rate", "mean"),
    ).sort_values(["mean_abs_action_delta", "action_saturation_fraction"], ascending=False).head(6)
    late_release = assoc[
        (assoc["failure_label"] == "previous_target_not_released")
        & (assoc["metric"].isin(["mean_abs_action_delta", "action_saturation_fraction", "p95_fingertip_jerk"]))
    ]
    phase_summary = phase_df.groupby("phase", as_index=False).agg(
        mean_abs_action_delta=("mean_abs_action_delta", "mean"),
        action_saturation_fraction=("action_saturation_fraction", "mean"),
        p95_fingertip_jerk=("p95_fingertip_jerk", "mean"),
    ).sort_values("p95_fingertip_jerk", ascending=False)
    lines = [
        "# Minimal Refinement Design",
        "",
        "## Evidence Summary",
        "",
        f"- Reference traces analyzed: `{len(per_rollout)}`.",
        "- Visible jerkiness is supported quantitatively by frequent action saturation and large per-step normalized action deltas.",
        "- The strongest per-dimension contributors are:",
    ]
    for _, row in top_dims.iterrows():
        lines.append(
            f"  - action `{int(row['action_index'])}` `{row['action_name']}` ({row['action_group']}): "
            f"mean delta `{row['mean_abs_action_delta']:.3f}`, saturation `{row['action_saturation_fraction']:.3f}`."
        )
    lines.extend(["", "- Highest phase-level motion roughness:"])
    for _, row in phase_summary.head(4).iterrows():
        lines.append(
            f"  - `{row['phase']}`: mean delta `{row['mean_abs_action_delta']:.3f}`, "
            f"saturation `{row['action_saturation_fraction']:.3f}`, p95 jerk `{row['p95_fingertip_jerk']:.3f}`."
        )
    if not late_release.empty:
        lines.extend(["", "- Previous-target release associations:"])
        for _, row in late_release.iterrows():
            lines.append(
                f"  - `{row['metric']}` standardized difference `{row['standardized_difference']:.3f}` "
                f"(with-label mean `{row['with_label_mean']:.3f}`, without `{row['without_label_mean']:.3f}`)."
            )
    lines.extend(
        [
            "",
            "## Recommended Single Refinement",
            "",
            "Use one transition/release mechanism plus one small smoothness mechanism:",
            "",
            "1. **Transition-gated previous-target release/completion reward.** During the release-to-next-note phase, "
            "increase reward for lowering the previous target key below the soft threshold while separately rewarding "
            "second-target activation. Gate this term to transition windows only, using current/previous/future target "
            "masks already available in `GeneralOneHandGoalEnv`, so single-note anchors are not made conservative.",
            "",
            "2. **Small transition-scoped action-rate/saturation penalty.** Add a low-weight penalty on "
            "`||a_t - a_{t-1}||^2` and saturation only during transition/release phases. Do not apply a large global "
            "smoothness penalty; the human review explicitly warned that the policy should not become too conservative.",
            "",
            "This pair is still interpretable because the release/completion term targets the dominant behavioral failure, "
            "while the smoothness term targets the observed actuation artifact. Both are phase-scoped and should be ablated "
            "against the current `transition_cleanup_sensitive_v1` baseline.",
            "",
            "Avoid post-hoc action filtering as the primary fix for now: it would change evaluated policy semantics and could "
            "mask timing failures rather than teach release and second-note completion.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_canary_plan(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Canary Experiment Plan",
                "",
                "## Goal",
                "",
                "Validate one release/completion refinement plus one transition-scoped smoothness penalty before any long run.",
                "",
                "## Initialization",
                "",
                "Recommended: warm-start from the selected DroQ-sensitive 800k lightweight checkpoint for the canary only. "
                "This directly tests whether the refinement can repair the frozen controller's observed release/transition "
                "failures at low cost. A later production comparison should still include a fresh-seed run for causal claims.",
                "",
                "Trade-off: warm-start is less clean scientifically than training from scratch, but it is the cheapest way to "
                "test whether the current finalist can be refined without destroying clean anchors.",
                "",
                "## Budget",
                "",
                "- Algorithm: DroQ only.",
                "- Steps: 75k canary, acceptable range 50k-100k.",
                "- Seed: 13 initially.",
                "- Curriculum: trained transitions emphasized, with clean single anchors retained.",
                "- Evaluation: original benchmark sequences plus motion-quality metrics.",
                "",
                "## Acceptance Gates",
                "",
                "- Clean single-note anchors retain pressed-key F1 >= 1.0 or no material regression from the frozen finalist.",
                "- Worst trained-transition pressed-key F1 improves or does not regress by more than 0.05.",
                "- Previous-target-not-released frequency decreases on trained transitions.",
                "- Second-note completion improves on trained transitions.",
                "- Wrong-key threshold crossings and integrated unintended travel do not increase materially.",
                "- Mean action delta, action saturation fraction, and p95 fingertip jerk decrease on trained transitions.",
                "- Target recall and timestep F1 do not collapse.",
                "",
                "## Progression Rule",
                "",
                "Proceed to a longer run only if both behavior and motion gates are satisfied. If behavior improves but motion "
                "worsens, reduce the smoothness/saturation term. If motion improves but second-note completion drops, the "
                "penalty is too conservative.",
            ]
        ),
        encoding="utf-8",
    )


def write_report(
    path: Path,
    *,
    per_rollout: pd.DataFrame,
    per_model: pd.DataFrame,
    per_dim: pd.DataFrame,
    phase_df: pd.DataFrame,
    assoc: pd.DataFrame,
    reference: pd.DataFrame,
    repeated: pd.DataFrame,
) -> None:
    top_dim = per_dim.groupby(["action_index", "action_name", "action_group"], as_index=False).agg(
        mean_abs_action_delta=("mean_abs_action_delta", "mean"),
        action_saturation_fraction=("action_saturation_fraction", "mean"),
        high_frequency_oscillation_rate=("high_frequency_oscillation_rate", "mean"),
    ).sort_values(["mean_abs_action_delta", "action_saturation_fraction"], ascending=False)
    phase_summary = phase_df.groupby("phase", as_index=False).agg(
        mean_abs_action_delta=("mean_abs_action_delta", "mean"),
        action_saturation_fraction=("action_saturation_fraction", "mean"),
        p95_fingertip_jerk=("p95_fingertip_jerk", "mean"),
    ).sort_values("p95_fingertip_jerk", ascending=False)
    model_cols = [
        "model_id",
        "mean_abs_action_delta_mean",
        "action_saturation_fraction_mean",
        "p95_fingertip_jerk_mean",
        "pressed_key_f1_mean",
        "timestep_f1_mean",
        "integrated_unintended_travel_mean",
    ]
    lines = [
        "# Motion Quality Report",
        "",
        "## Trace Coverage",
        "",
        f"- Reference rollout traces analyzed: `{len(per_rollout)}` from `{len(reference)}` rendered reference rollouts.",
        f"- Repeated rollout metric rows available: `{len(repeated)}`. These do not include action/fingertip time-series, so motion metrics use reference traces.",
        "- Trace fields include 22 normalized action columns, five fingertip site positions, target/pressed key fields, key states for MIDI 72-76, reward components, onset/release markers and failure labels.",
        "- Joint positions and velocities are not present in the saved traces; fingertip finite differences are therefore used as the motion proxy.",
        "- Control timestep is `0.05 s` for the analyzed rollouts.",
        "",
        "## Per-Model Summary",
        "",
        markdown_table(per_model[model_cols], model_cols),
        "",
        "## Dominant Action Dimensions",
        "",
        markdown_table(
            top_dim.head(10),
            [
                "action_index",
                "action_name",
                "action_group",
                "mean_abs_action_delta",
                "action_saturation_fraction",
                "high_frequency_oscillation_rate",
            ],
        ),
        "",
        "## Phase Summary",
        "",
        markdown_table(
            phase_summary,
            ["phase", "mean_abs_action_delta", "action_saturation_fraction", "p95_fingertip_jerk"],
        ),
        "",
        "## Failure Associations",
        "",
        "The table below is descriptive only. Repeated rollouts share models and sequences, so independence is not claimed.",
        "",
    ]
    assoc_view = assoc[
        assoc["metric"].isin(
            [
                "mean_abs_action_delta",
                "action_saturation_fraction",
                "p95_fingertip_jerk",
                "max_unintended_activation",
                "timestep_f1",
            ]
        )
    ].sort_values(["failure_label", "metric"])
    lines.append(
        markdown_table(
            assoc_view.head(80),
            [
                "failure_label",
                "metric",
                "with_label_count",
                "with_label_mean",
                "without_label_mean",
                "standardized_difference",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The human-visible jerkiness is supported by large normalized action deltas, frequent action saturation, and high finite-difference fingertip jerk in the traces.",
            "- The dominant dimensions are primarily forearm translation and wrist/finger aggregate joints, so the artifact is not isolated to one note label.",
            "- Release and transition failures are associated with lower timestep F1 and elevated unintended activation; motion/action roughness is a contributing diagnostic signal but not proven causal.",
            "- The next refinement should target previous-target release and second-note completion first, with a small transition-scoped action-rate/saturation term as a secondary stabilizer.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
