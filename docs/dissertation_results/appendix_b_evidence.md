# Appendix B Evidence Summary

This file is an evidence-only collation from existing local result CSV/JSON files. Source experiment outputs were not modified and no new training/evaluation was run. Stored metrics are reported directly where available. Values labelled as calculated are straightforward aggregations or correlations from named raw rows.

Numeric values are printed to at least six decimal places where applicable.

## 1. Pipeline 1 Controller-Selection Factorial

Source: `artifacts/five_note_factorial_1m_hex_export/analysis/per_condition_comparison.csv` and `artifacts/five_note_factorial_1m_hex_export/analysis/best_checkpoint_per_condition.csv`.

| condition_id | algorithm | reward_profile | selected_checkpoint_step | overall_rank | selected_gate_anchor_clean | selected_trained_pressed_key_precision | selected_trained_pressed_key_recall | selected_trained_pressed_key_f1 | selected_trained_timestep_f1 | selected_integrated_unintended | trained_max_unintended_mean | selected_wrong_key_press_count | selected_heldout_pressed_key_f1 | heldout_timestep_f1_mean | selected_composition_pressed_key_f1 | composition_timestep_f1_mean | single_f1_mean | transition_f1_mean | selection_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| droq_sensitive_v1 | DroQ | transition_cleanup_sensitive_v1 | 800000 | 1 | 1 | 0.935897 | 0.730769 | 0.792308 | 0.684639 | 1.958102 | 0.697350 | 0.153846 | 0.566667 | 0.471582 | 0.472222 | 0.303367 | 1.000000 | 0.662500 | best balanced clean-anchor checkpoint: high trained F1/timestep F1 with low wrong presses and lower unintended travel than 1M |
| sac_sensitive_v1 | SAC | transition_cleanup_sensitive_v1 | 700000 | 2 | 1 | 1.000000 | 0.692308 | 0.794872 | 0.363122 | 0.355578 | 0.244361 | 0.000000 | 0.666667 | 0.284115 | 0.472222 | 0.182057 | 1.000000 | 0.666667 | cleanest reliable-anchor SAC checkpoint with zero wrong trained presses; misses many second transition notes |
| sac_original | SAC | transition_cleanup | 900000 | 3 | 0 | 0.820513 | 0.807692 | 0.779487 | 0.758906 | 4.578125 | 0.856186 | 0.538462 | 0.483069 | 0.438369 | 0.440608 | 0.355679 | 0.933333 | 0.683333 | best timing/accuracy checkpoint, but dirty anchors and high unintended travel |
| droq_original | DroQ | transition_cleanup | 700000 | 4 | 0 | 0.782051 | 0.846154 | 0.792308 | 0.646785 | 3.245236 | 0.869848 | 0.538462 | 0.491667 | 0.423405 | 0.501599 | 0.325974 | 0.866667 | 0.745833 | best original-DroQ target/timing checkpoint, but fails clean-anchor gate |

Selected canonical controller: `droq_sensitive_v1` at checkpoint `800000`, from the stored `overall_rank=1` row and selection reason in `per_condition_comparison.csv`.

## 2. Pipeline 2 Phase-A

### Checkpoint-level pressed-key F1 (correct audio)

| seed | checkpoint_step | pressed_key_f1_mean | source |
| --- | --- | --- | --- |
| 13 | 10000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 13 | 25000 | 0.358974 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 13 | 50000 | 0.794872 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 13 | 100000 | 0.794872 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 13 | 250000 | 0.794872 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 13 | 500000 | 0.830769 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 13 | 750000 | 0.728205 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 13 | 1000000 | 0.846154 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | 10000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | 25000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | 50000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | 100000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | 250000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | 500000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | 750000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | 1000000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | 10000 | 0.397436 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | 25000 | 0.487179 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | 50000 | 0.615385 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | 100000 | 0.705128 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | 250000 | 0.769231 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | 500000 | 0.782051 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | 750000 | 0.792308 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | 1000000 | 0.794872 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |

### 1M aggregate metrics (correct audio)

| seed | checkpoint_step | precision | recall | pressed_key_f1 | timestep_f1 | integrated_unintended | max_unintended_mean | max_unintended_max | wrong_press_count_mean | wrong_press_count_total | anchor_f1_mean | anchor_f1_min | transition_f1_mean | transition_f1_min | transition_timestep_f1_mean | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | 1000000 | 1.000000 | 0.769231 | 0.846154 | 0.652103 | 0.017429 | 0.231828 | 0.383005 | 0.000000 | 0 | 1.000000 | 1.000000 | 0.750000 | 0.666667 | 0.571032 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | 1000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.019873 | 0.397460 | 0.469725 | 0.000000 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | 1000000 | 0.884615 | 0.769231 | 0.794872 | 0.543438 | 0.044280 | 0.595916 | 1.000000 | 0.230769 | 3 | 0.933333 | 0.666667 | 0.708333 | 0.500000 | 0.466421 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |

### 1M correct/zero/mismatched audio aggregate metrics

| seed | audio_mode | pressed_key_f1 | precision | recall | timestep_f1 | integrated_unintended | max_unintended_mean | wrong_press_count_mean | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | correct | 0.846154 | 1.000000 | 0.769231 | 0.652103 | 0.017429 | 0.231828 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 13 | mismatched | 0.384615 | 0.538462 | 0.307692 | 0.105983 | 0.206077 | 0.818302 | 0.538462 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 13 | zero | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | correct | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.019873 | 0.397460 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | mismatched | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.019873 | 0.397460 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 37 | zero | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.019873 | 0.397460 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | correct | 0.794872 | 0.884615 | 0.769231 | 0.543438 | 0.044280 | 0.595916 | 0.230769 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | mismatched | 0.346154 | 0.423077 | 0.307692 | 0.073970 | 0.188535 | 0.984721 | 0.769231 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |
| 61 | zero | 0.333333 | 0.269231 | 0.461538 | 0.100000 | 0.298508 | 0.971587 | 2.000000 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv |

### 1M audio-action dependence summaries (calculated from stored rows)

| seed | n | mean_abs_action_diff_correct_zero_mean | mean_abs_action_diff_correct_mismatch_mean | max_abs_action_diff_correct_zero_max | max_abs_action_diff_correct_mismatch_max | source |
| --- | --- | --- | --- | --- | --- | --- |
| 13 | 13 | 0.573785 | 0.437901 | 1.787470 | 1.807417 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/audio_dependence.csv |
| 37 | 13 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/audio_dependence.csv |
| 61 | 13 | 0.338708 | 0.330034 | 1.669416 | 1.963123 | artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/audio_dependence.csv |

## 3. Long-Horizon Benchmark

Source: `artifacts/paironly_final_evaluation/analysis_v1/long_horizon_by_length.csv`. Rows are stored length-level summaries. Pipeline 2 seed61 long-horizon rows were not found locally; see missing-results section.

| pipeline_family | model_variant | condition | benchmark | sequence_length | n | timestep_f1 | event_hit | pressed_key_f1 | max_unintended | integrated_unintended |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline1 | p1_base | basic_pitch | frozen_long_horizon_compositional | 3 | 5 | 0.425714 | 0.466667 | 0.621905 | 1.000000 | 0.233937 |
| pipeline1 | p1_base | basic_pitch | frozen_long_horizon_compositional | 5 | 5 | 0.277960 | 0.360000 | 0.538889 | 1.000000 | 0.358029 |
| pipeline1 | p1_base | basic_pitch | frozen_long_horizon_compositional | 10 | 5 | 0.163513 | 0.180000 | 0.505556 | 1.000000 | 0.358029 |
| pipeline1 | p1_base | basic_pitch | frozen_long_horizon_compositional | 20 | 5 | 0.089869 | 0.090000 | 0.488889 | 1.000000 | 0.358029 |
| pipeline1 | p1_base | oracle | frozen_long_horizon_compositional | 3 | 5 | 0.392275 | 0.333333 | 0.414286 | 0.947870 | 0.160877 |
| pipeline1 | p1_base | oracle | frozen_long_horizon_compositional | 5 | 5 | 0.286051 | 0.200000 | 0.401587 | 0.801028 | 0.171882 |
| pipeline1 | p1_base | oracle | frozen_long_horizon_compositional | 10 | 5 | 0.156698 | 0.100000 | 0.368254 | 0.801028 | 0.171882 |
| pipeline1 | p1_base | oracle | frozen_long_horizon_compositional | 20 | 5 | 0.082294 | 0.050000 | 0.334921 | 0.801028 | 0.171882 |
| pipeline2 | p2_base | correct | frozen_long_horizon_compositional | 3 | 5 | 0.445475 | 0.466667 | 0.700952 | 0.532917 | 0.040143 |
| pipeline2 | p2_base | correct | frozen_long_horizon_compositional | 5 | 5 | 0.336614 | 0.440000 | 0.690952 | 0.876514 | 0.132275 |
| pipeline2 | p2_base | correct | frozen_long_horizon_compositional | 10 | 5 | 0.324969 | 0.400000 | 0.666667 | 0.992822 | 0.308231 |
| pipeline2 | p2_base | correct | frozen_long_horizon_compositional | 20 | 5 | 0.234994 | 0.320000 | 0.835556 | 1.000000 | 0.732824 |

### Correlations (calculated from stored per-sequence rows)

| pipeline_family | model_variant | condition | n | sequence_length_vs_timestep_f1_pearson | transition_novelty_vs_timestep_f1_pearson | source |
| --- | --- | --- | --- | --- | --- | --- |
| pipeline1 | p1_base | basic_pitch | 20 | -0.886104 | MISSING: no stored numeric transition-novelty column found | artifacts/paironly_final_evaluation/analysis_v1/sequence_level_master.csv |
| pipeline1 | p1_base | oracle | 20 | -0.904474 | MISSING: no stored numeric transition-novelty column found | artifacts/paironly_final_evaluation/analysis_v1/sequence_level_master.csv |
| pipeline2 | p2_base | correct | 20 | -0.567008 | MISSING: no stored numeric transition-novelty column found | artifacts/paironly_final_evaluation/analysis_v1/sequence_level_master.csv |

Per-sequence rows are available in `artifacts/paironly_final_evaluation/analysis_v1/sequence_level_master.csv`; the full rows are not pasted here to keep the appendix evidence compact.

## 4. Pair+Horizon Adaptation

Source: `artifacts/paironly_final_evaluation/analysis_v1/headline_metrics.csv`. Stored evaluated variants are base and final pair+horizon adaptation. Intermediate 100k/250k per-checkpoint evaluation rows were not found in the located analysis outputs.

| pipeline_family | model_variant | benchmark | condition | sequence_count | pressed_key_precision_mean | pressed_key_recall_mean | pressed_key_f1_mean | timestep_f1_mean | target_event_hit_rate_mean | ordered_event_accuracy_mean | transition_event_pair_accuracy_mean | max_unintended_key_state_mean | integrated_unintended_key_state_mean | wrong_press_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline1 | p1_base | clean_unseen_compositions | basic_pitch | 8 | 0.604167 | 0.283333 | 0.381250 | 0.236133 | 0.183333 | 0.183333 | 0.013889 | 0.978870 | 0.246073 | 0.875000 |
| pipeline1 | p1_base | clean_unseen_compositions | oracle | 8 | 0.937500 | 0.350000 | 0.489286 | 0.346571 | 0.256250 | 0.256250 | 0.114218 | 0.680594 | 0.121454 | 0.125000 |
| pipeline1 | p1_base | complete_pairwise | basic_pitch | 30 | 0.916667 | 0.683333 | 0.750000 | 0.433626 | 0.583333 | 0.583333 | 0.166667 | 0.345055 | 0.021817 | 0.166667 |
| pipeline1 | p1_base | complete_pairwise | oracle | 30 | 0.916667 | 0.683333 | 0.750000 | 0.433626 | 0.583333 | 0.583333 | 0.166667 | 0.345055 | 0.021817 | 0.166667 |
| pipeline1 | p1_base | exact_13_retention | basic_pitch | 13 | 0.923077 | 0.692308 | 0.756410 | 0.482253 | 0.692308 | 0.692308 | 0.384615 | 0.289793 | 0.018001 | 0.153846 |
| pipeline1 | p1_base | exact_13_retention | oracle | 13 | 0.923077 | 0.692308 | 0.756410 | 0.482253 | 0.692308 | 0.692308 | 0.384615 | 0.289793 | 0.018001 | 0.153846 |
| pipeline1 | p1_base | extrapolation_30_40 | basic_pitch | 10 | 0.540476 | 0.640000 | 0.577622 | 0.052355 | 0.070000 | 0.052500 | 0.024050 | 1.000000 | 0.612848 | 3.000000 |
| pipeline1 | p1_base | extrapolation_30_40 | oracle | 10 | 0.666667 | 0.360000 | 0.460317 | 0.072178 | 0.052500 | 0.046667 | 0.018037 | 0.975358 | 0.277773 | 1.000000 |
| pipeline1 | p1_base | frozen_long_horizon_compositional | basic_pitch | 20 | 0.675000 | 0.530000 | 0.538810 | 0.239264 | 0.274167 | 0.274167 | 0.132749 | 1.000000 | 0.327006 | 1.600000 |
| pipeline1 | p1_base | frozen_long_horizon_compositional | oracle | 20 | 0.650000 | 0.300000 | 0.379762 | 0.229330 | 0.170833 | 0.170833 | 0.000000 | 0.837739 | 0.169131 | 1.200000 |
| pipeline1 | p1_pairhorizon | clean_unseen_compositions | basic_pitch | 8 | 1.000000 | 0.233333 | 0.375000 | 0.054774 | 0.170833 | 0.170833 | 0.000000 | 0.024601 | 0.001230 | 0.000000 |
| pipeline1 | p1_pairhorizon | clean_unseen_compositions | oracle | 8 | 1.000000 | 0.233333 | 0.375000 | 0.054774 | 0.170833 | 0.170833 | 0.000000 | 0.024601 | 0.001230 | 0.000000 |
| pipeline1 | p1_pairhorizon | complete_pairwise | basic_pitch | 30 | 1.000000 | 0.666667 | 0.777778 | 0.204945 | 0.583333 | 0.583333 | 0.166667 | 0.059834 | 0.003110 | 0.000000 |
| pipeline1 | p1_pairhorizon | complete_pairwise | oracle | 30 | 1.000000 | 0.666667 | 0.777778 | 0.204945 | 0.583333 | 0.583333 | 0.166667 | 0.059834 | 0.002992 | 0.000000 |
| pipeline1 | p1_pairhorizon | exact_13_retention | basic_pitch | 13 | 1.000000 | 0.692308 | 0.794872 | 0.231192 | 0.692308 | 0.692308 | 0.384615 | 0.060535 | 0.003117 | 0.000000 |
| pipeline1 | p1_pairhorizon | exact_13_retention | oracle | 13 | 1.000000 | 0.692308 | 0.794872 | 0.231192 | 0.692308 | 0.692308 | 0.384615 | 0.060535 | 0.003027 | 0.000000 |
| pipeline1 | p1_pairhorizon | extrapolation_30_40 | basic_pitch | 10 | 1.000000 | 0.200000 | 0.333333 | 0.011590 | 0.029167 | 0.029167 | 0.000000 | 0.210697 | 0.013598 | 0.000000 |
| pipeline1 | p1_pairhorizon | extrapolation_30_40 | oracle | 10 | 1.000000 | 0.200000 | 0.333333 | 0.011590 | 0.029167 | 0.029167 | 0.000000 | 0.249359 | 0.017165 | 0.000000 |
| pipeline1 | p1_pairhorizon | frozen_long_horizon_compositional | basic_pitch | 20 | 1.000000 | 0.270000 | 0.416667 | 0.069671 | 0.170833 | 0.170833 | 0.000000 | 0.124978 | 0.007621 | 0.000000 |
| pipeline1 | p1_pairhorizon | frozen_long_horizon_compositional | oracle | 20 | 1.000000 | 0.270000 | 0.416667 | 0.069671 | 0.170833 | 0.170833 | 0.000000 | 0.124978 | 0.007621 | 0.000000 |
| pipeline2 | p2_base | clean_unseen_compositions | correct | 8 | 0.864583 | 0.408333 | 0.540675 | 0.268680 | 0.339583 | 0.170833 | 0.013889 | 0.806667 | 0.259707 | 0.500000 |
| pipeline2 | p2_base | complete_pairwise | correct | 30 | 0.958333 | 0.716667 | 0.794444 | 0.536169 | 0.650000 | 0.650000 | 0.300000 | 0.322872 | 0.028108 | 0.133333 |
| pipeline2 | p2_base | exact_13_retention | correct | 13 | 1.000000 | 0.769231 | 0.846154 | 0.671999 | 0.769231 | 0.769231 | 0.538462 | 0.221166 | 0.015745 | 0.000000 |
| pipeline2 | p2_base | extrapolation_30_40 | correct | 10 | 0.831429 | 0.780000 | 0.795859 | 0.230008 | 0.254167 | 0.029167 | 0.019717 | 1.000000 | 0.822425 | 0.900000 |
| pipeline2 | p2_base | frozen_long_horizon_compositional | correct | 20 | 0.854167 | 0.656667 | 0.723532 | 0.335513 | 0.406667 | 0.205000 | 0.109649 | 0.850563 | 0.303368 | 0.600000 |
| pipeline2 | p2_pairhorizon | clean_unseen_compositions | correct | 8 | 1.000000 | 0.483333 | 0.637897 | 0.346858 | 0.358333 | 0.170833 | 0.044408 | 0.918429 | 0.331573 | 0.000000 |
| pipeline2 | p2_pairhorizon | complete_pairwise | correct | 30 | 0.983333 | 0.733333 | 0.811111 | 0.519742 | 0.700000 | 0.700000 | 0.400000 | 0.553573 | 0.070916 | 0.033333 |
| pipeline2 | p2_pairhorizon | exact_13_retention | correct | 13 | 1.000000 | 0.730769 | 0.820513 | 0.599727 | 0.769231 | 0.769231 | 0.538462 | 0.545359 | 0.073286 | 0.000000 |
| pipeline2 | p2_pairhorizon | extrapolation_30_40 | correct | 10 | 1.000000 | 0.600000 | 0.738095 | 0.363065 | 0.290833 | 0.029167 | 0.069496 | 1.000000 | 1.344376 | 0.000000 |
| pipeline2 | p2_pairhorizon | frozen_long_horizon_compositional | correct | 20 | 0.966667 | 0.416667 | 0.568373 | 0.351119 | 0.361667 | 0.205000 | 0.050950 | 0.826441 | 0.299350 | 0.100000 |

### Earlier refined pair+long-horizon evaluation overall rows

Source: `artifacts/paircomplete_refined_evaluation/analysis_v1/headline_metrics.csv`.

| benchmark | model | audio_mode | sequence_count | pressed_key_precision | pressed_key_recall | pressed_key_f1 | timestep_f1 | target_event_hit_rate | ordered_event_accuracy | transition_event_pair_accuracy | max_unintended_key_state | integrated_unintended_key_state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| complete_pairwise | P1 Oracle refined | oracle | 30 | 1.000000 | 0.666667 | 0.777778 | 0.204945 | 0.583333 | 0.583333 | 0.166667 | 0.059834 | 0.002992 |
| complete_pairwise | P2 direct refined | correct | 30 | 0.983333 | 0.733333 | 0.811111 | 0.519742 | 0.700000 | 0.700000 | 0.400000 | 0.553573 | 0.070916 |
| frozen_long_horizon_v1 | P1 Oracle refined | oracle | 20 | 1.000000 | 0.270000 | 0.416667 | 0.069671 | 0.170833 | 0.170833 | 0.000000 | 0.124978 | 0.007621 |
| frozen_long_horizon_v1 | P2 direct refined | correct | 20 | 0.966667 | 0.416667 | 0.568373 | 0.351119 | 0.361667 | 0.205000 | 0.050950 | 0.826441 | 0.299350 |
| validation_compositions | P1 Oracle refined | oracle | 8 | 1.000000 | 0.233333 | 0.375000 | 0.080175 | 0.170833 | 0.170833 | 0.000000 | 0.190914 | 0.012290 |
| validation_compositions | P2 direct refined | correct | 8 | 1.000000 | 0.500000 | 0.658036 | 0.382994 | 0.406250 | 0.225000 | 0.128107 | 0.909232 | 0.307649 |
| clean_composition_test | P1 Oracle refined | oracle | 8 | 1.000000 | 0.233333 | 0.375000 | 0.054774 | 0.170833 | 0.170833 | 0.000000 | 0.024601 | 0.001230 |
| clean_composition_test | P2 direct refined | correct | 8 | 1.000000 | 0.483333 | 0.637897 | 0.346858 | 0.358333 | 0.170833 | 0.044408 | 0.918429 | 0.331573 |
| extrapolation_30_40 | P1 Oracle refined | oracle | 10 | 1.000000 | 0.200000 | 0.333333 | 0.011590 | 0.029167 | 0.029167 | 0.000000 | 0.249359 | 0.017165 |
| extrapolation_30_40 | P2 direct refined | correct | 10 | 1.000000 | 0.600000 | 0.738095 | 0.363065 | 0.290833 | 0.029167 | 0.069496 | 1.000000 | 1.344376 |

## 5. Pair-Only Adaptation

Source: `artifacts/paironly_final_evaluation/analysis_v1/headline_metrics.csv`. Stored evaluated variants are base and final pair-only adaptation. Intermediate 100k/250k per-checkpoint evaluation rows were not found in the located analysis outputs.

| pipeline_family | model_variant | benchmark | condition | sequence_count | pressed_key_precision_mean | pressed_key_recall_mean | pressed_key_f1_mean | timestep_f1_mean | target_event_hit_rate_mean | ordered_event_accuracy_mean | transition_event_pair_accuracy_mean | max_unintended_key_state_mean | integrated_unintended_key_state_mean | wrong_press_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline1 | p1_base | clean_unseen_compositions | basic_pitch | 8 | 0.604167 | 0.283333 | 0.381250 | 0.236133 | 0.183333 | 0.183333 | 0.013889 | 0.978870 | 0.246073 | 0.875000 |
| pipeline1 | p1_base | clean_unseen_compositions | oracle | 8 | 0.937500 | 0.350000 | 0.489286 | 0.346571 | 0.256250 | 0.256250 | 0.114218 | 0.680594 | 0.121454 | 0.125000 |
| pipeline1 | p1_base | complete_pairwise | basic_pitch | 30 | 0.916667 | 0.683333 | 0.750000 | 0.433626 | 0.583333 | 0.583333 | 0.166667 | 0.345055 | 0.021817 | 0.166667 |
| pipeline1 | p1_base | complete_pairwise | oracle | 30 | 0.916667 | 0.683333 | 0.750000 | 0.433626 | 0.583333 | 0.583333 | 0.166667 | 0.345055 | 0.021817 | 0.166667 |
| pipeline1 | p1_base | exact_13_retention | basic_pitch | 13 | 0.923077 | 0.692308 | 0.756410 | 0.482253 | 0.692308 | 0.692308 | 0.384615 | 0.289793 | 0.018001 | 0.153846 |
| pipeline1 | p1_base | exact_13_retention | oracle | 13 | 0.923077 | 0.692308 | 0.756410 | 0.482253 | 0.692308 | 0.692308 | 0.384615 | 0.289793 | 0.018001 | 0.153846 |
| pipeline1 | p1_base | extrapolation_30_40 | basic_pitch | 10 | 0.540476 | 0.640000 | 0.577622 | 0.052355 | 0.070000 | 0.052500 | 0.024050 | 1.000000 | 0.612848 | 3.000000 |
| pipeline1 | p1_base | extrapolation_30_40 | oracle | 10 | 0.666667 | 0.360000 | 0.460317 | 0.072178 | 0.052500 | 0.046667 | 0.018037 | 0.975358 | 0.277773 | 1.000000 |
| pipeline1 | p1_base | frozen_long_horizon_compositional | basic_pitch | 20 | 0.675000 | 0.530000 | 0.538810 | 0.239264 | 0.274167 | 0.274167 | 0.132749 | 1.000000 | 0.327006 | 1.600000 |
| pipeline1 | p1_base | frozen_long_horizon_compositional | oracle | 20 | 0.650000 | 0.300000 | 0.379762 | 0.229330 | 0.170833 | 0.170833 | 0.000000 | 0.837739 | 0.169131 | 1.200000 |
| pipeline1 | p1_paironly | clean_unseen_compositions | basic_pitch | 8 | 0.631250 | 0.508333 | 0.533117 | 0.149787 | 0.258333 | 0.183333 | 0.013889 | 1.000000 | 0.386959 | 1.750000 |
| pipeline1 | p1_paironly | clean_unseen_compositions | oracle | 8 | 0.645833 | 0.400000 | 0.473611 | 0.120898 | 0.214583 | 0.202083 | 0.037829 | 0.989249 | 0.255108 | 1.125000 |
| pipeline1 | p1_paironly | complete_pairwise | basic_pitch | 30 | 0.966667 | 0.683333 | 0.772222 | 0.472325 | 0.583333 | 0.583333 | 0.166667 | 0.293014 | 0.020743 | 0.066667 |
| pipeline1 | p1_paironly | complete_pairwise | oracle | 30 | 1.000000 | 0.666667 | 0.777778 | 0.445024 | 0.583333 | 0.583333 | 0.166667 | 0.053626 | 0.002681 | 0.000000 |
| pipeline1 | p1_paironly | exact_13_retention | basic_pitch | 13 | 0.961538 | 0.692308 | 0.769231 | 0.491378 | 0.692308 | 0.692308 | 0.384615 | 0.215545 | 0.014411 | 0.076923 |
| pipeline1 | p1_paironly | exact_13_retention | oracle | 13 | 1.000000 | 0.692308 | 0.794872 | 0.471243 | 0.692308 | 0.692308 | 0.384615 | 0.046277 | 0.002314 | 0.000000 |
| pipeline1 | p1_paironly | extrapolation_30_40 | basic_pitch | 10 | 0.260498 | 0.400000 | 0.311830 | 0.031147 | 0.058333 | 0.029167 | 0.006012 | 1.000000 | 0.703842 | 5.800000 |
| pipeline1 | p1_paironly | extrapolation_30_40 | oracle | 10 | 0.399048 | 0.480000 | 0.427330 | 0.027604 | 0.064167 | 0.029167 | 0.006012 | 1.000000 | 0.590113 | 3.800000 |
| pipeline1 | p1_paironly | frozen_long_horizon_compositional | basic_pitch | 20 | 0.745833 | 0.330000 | 0.407771 | 0.174967 | 0.188333 | 0.188333 | 0.020687 | 1.000000 | 0.286634 | 0.950000 |
| pipeline1 | p1_paironly | frozen_long_horizon_compositional | oracle | 20 | 0.712500 | 0.441667 | 0.496104 | 0.162678 | 0.256667 | 0.221667 | 0.091374 | 0.942408 | 0.274416 | 1.100000 |
| pipeline2 | p2_base | clean_unseen_compositions | correct | 8 | 0.864583 | 0.408333 | 0.540675 | 0.268680 | 0.339583 | 0.170833 | 0.013889 | 0.806667 | 0.259707 | 0.500000 |
| pipeline2 | p2_base | complete_pairwise | correct | 30 | 0.958333 | 0.716667 | 0.794444 | 0.536169 | 0.650000 | 0.650000 | 0.300000 | 0.322872 | 0.028108 | 0.133333 |
| pipeline2 | p2_base | exact_13_retention | correct | 13 | 1.000000 | 0.769231 | 0.846154 | 0.671999 | 0.769231 | 0.769231 | 0.538462 | 0.221166 | 0.015745 | 0.000000 |
| pipeline2 | p2_base | extrapolation_30_40 | correct | 10 | 0.831429 | 0.780000 | 0.795859 | 0.230008 | 0.254167 | 0.029167 | 0.019717 | 1.000000 | 0.822425 | 0.900000 |
| pipeline2 | p2_base | frozen_long_horizon_compositional | correct | 20 | 0.854167 | 0.656667 | 0.723532 | 0.335513 | 0.406667 | 0.205000 | 0.109649 | 0.850563 | 0.303368 | 0.600000 |
| pipeline2 | p2_paironly | clean_unseen_compositions | correct | 8 | 0.640972 | 0.766667 | 0.645185 | 0.265608 | 0.431250 | 0.256250 | 0.147113 | 0.892332 | 0.658650 | 2.375000 |
| pipeline2 | p2_paironly | complete_pairwise | correct | 30 | 0.863333 | 0.683333 | 0.735714 | 0.634003 | 0.666667 | 0.633333 | 0.466667 | 0.559289 | 0.088578 | 0.233333 |
| pipeline2 | p2_paironly | exact_13_retention | correct | 13 | 0.807692 | 0.653846 | 0.705128 | 0.655789 | 0.653846 | 0.615385 | 0.538462 | 0.564683 | 0.066697 | 0.230769 |
| pipeline2 | p2_paironly | extrapolation_30_40 | correct | 10 | 0.540159 | 0.780000 | 0.634509 | 0.250305 | 0.290000 | 0.023333 | 0.073828 | 1.000000 | 1.541713 | 3.400000 |
| pipeline2 | p2_paironly | frozen_long_horizon_compositional | correct | 20 | 0.697460 | 0.518333 | 0.551548 | 0.360609 | 0.367500 | 0.170833 | 0.072588 | 0.792572 | 0.414730 | 1.450000 |

### Complete all-25 pairwise summaries

Source: `artifacts/paironly_final_evaluation/analysis_v1/pairwise_all25_summary.csv`.

| pipeline_family | model_variant | condition | n | both_events_activated | strict_successes | event_hit_rate_mean | ordered_event_accuracy_mean | transition_pair_accuracy_mean | timestep_f1_mean | pressed_f1_mean | max_unintended_mean | integrated_unintended_mean | wrong_press_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline1 | p1_base | basic_pitch | 25 | 0 | 0 | 0.500000 | 0.500000 | 0.000000 | 0.391261 | 0.713333 | 0.345055 | 0.021817 | 0.160000 |
| pipeline1 | p1_base | oracle | 25 | 0 | 0 | 0.500000 | 0.500000 | 0.000000 | 0.391261 | 0.713333 | 0.345055 | 0.021817 | 0.160000 |
| pipeline1 | p1_pairhorizon | basic_pitch | 25 | 0 | 0 | 0.500000 | 0.500000 | 0.000000 | 0.180220 | 0.733333 | 0.059834 | 0.003110 | 0.000000 |
| pipeline1 | p1_pairhorizon | oracle | 25 | 0 | 0 | 0.500000 | 0.500000 | 0.000000 | 0.180220 | 0.733333 | 0.059834 | 0.002992 | 0.000000 |
| pipeline1 | p1_paironly | basic_pitch | 25 | 0 | 0 | 0.500000 | 0.500000 | 0.000000 | 0.431772 | 0.740000 | 0.278532 | 0.019985 | 0.040000 |
| pipeline1 | p1_paironly | oracle | 25 | 0 | 0 | 0.500000 | 0.500000 | 0.000000 | 0.403206 | 0.733333 | 0.053626 | 0.002681 | 0.000000 |
| pipeline2 | p2_base | correct | 25 | 4 | 4 | 0.580000 | 0.580000 | 0.160000 | 0.487039 | 0.753333 | 0.340884 | 0.030167 | 0.160000 |
| pipeline2 | p2_pairhorizon | correct | 25 | 7 | 1 | 0.640000 | 0.640000 | 0.280000 | 0.472261 | 0.773333 | 0.590208 | 0.079721 | 0.040000 |
| pipeline2 | p2_paironly | correct | 25 | 9 | 4 | 0.640000 | 0.600000 | 0.360000 | 0.600804 | 0.722857 | 0.560957 | 0.096444 | 0.240000 |

## 6. Recorded-Audio Distribution Shift

### Overall recorded-audio headline metrics

Source: `artifacts/real_audio_distribution_shift/analysis_v1/headline_metrics.csv`.

| pipeline | model_label | condition_group | audio_mode | sequence_count | pressed_key_f1 | timestep_f1 | event_hit | ordered_event | max_unintended | integrated_unintended |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline1 | Pipeline1 Basic Pitch | evaluation_v1 | basic_pitch | 114 | 0.621166 | 0.475445 | 0.583333 | 0.583333 | 0.999809 | 0.325797 |
| pipeline1 | Pipeline1 Oracle | evaluation_v1 | oracle | 114 | 0.632832 | 0.594868 | 0.593421 | 0.593421 | 0.705787 | 0.124411 |
| pipeline2 | Pipeline2 seed13 1M real-audio | evaluation_v1 | correct | 114 | 0.276431 | 0.069773 | 0.382749 | 0.256579 | 0.988760 | 0.740364 |
| pipeline2 | Pipeline2 seed13 1M real-audio | evaluation_v1 | mismatched | 114 | 0.281643 | 0.079158 | 0.366520 | 0.236257 | 0.990533 | 0.701018 |
| pipeline2 | Pipeline2 seed13 1M real-audio | evaluation_v1 | zero | 114 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### Basic Pitch transcription metrics

Source: `artifacts/real_audio_distribution_shift/analysis_v1/basic_pitch_summary_interpretable.csv`. Strict precision/recall are calculated directly from stored `strict_matched_notes`, `predicted_notes`, and `expected_notes`; `pitch_presence_f1` is stored, but separate pitch-presence precision/recall were not found.

| condition | benchmark | sequence_rows | expected_notes | predicted_notes | strict_matched_notes | strict_note_precision_calculated | strict_note_recall_calculated | strict_note_f1 | pitch_presence_f1 | duplicate_or_spurious_notes | onset_mae | offset_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real_basic_pitch | clean_unseen_compositions | 24 | 228 | 458 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.956481 | 357.000000 | NA | NA |
| real_basic_pitch | complete_pairwise | 90 | 165 | 201 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.914815 | 74.000000 | NA | NA |
| synthetic_basic_pitch | clean_unseen_compositions | 8 | 76 | 84 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 48.000000 | NA | NA |
| synthetic_basic_pitch | complete_pairwise | 30 | 55 | 86 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 36.000000 | NA | NA |
| synthetic_basic_pitch | exact_13_retention | 13 | 21 | 35 | 0 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 14.000000 | NA | NA |

### Recorded metrics by pairwise vs clean unseen-composition sets

Source: `artifacts/real_audio_distribution_shift/analysis_v1/real_audio_headline_metrics.csv`.

| pipeline | model_variant | condition | benchmark | target_event_hit_rate | pressed_key_f1 | timestep_f1 | ordered_event_accuracy | transition_event_pair_accuracy | max_unintended_key_state | integrated_unintended_key_state | wrong_press_count | unintended_press_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline1 | p1_base_basic_pitch | real_basic_pitch | clean_unseen_compositions | 0.270833 | 0.508013 | 0.238902 | 0.270833 | 0.131457 | 1.000000 | 0.418846 | 2.583333 | 5.708333 |
| pipeline1 | p1_base_basic_pitch | real_basic_pitch | complete_pairwise | 0.666667 | 0.651340 | 0.538523 | 0.666667 | 0.333333 | 0.999758 | 0.300984 | 1.366667 | 2.833333 |
| pipeline1 | p1_base_oracle | real_oracle | clean_unseen_compositions | 0.256250 | 0.489286 | 0.346571 | 0.256250 | 0.114218 | 0.680595 | 0.121464 | 0.125000 | 0.125000 |
| pipeline1 | p1_base_oracle | real_oracle | complete_pairwise | 0.683333 | 0.671111 | 0.661081 | 0.683333 | 0.366667 | 0.712505 | 0.125197 | 0.866667 | 1.000000 |
| pipeline2 | p2_base | real_correct | clean_unseen_compositions | 0.401389 | 0.429334 | 0.061622 | 0.052083 | 0.170504 | 1.000000 | 2.239570 | 9.166667 | 35.208333 |
| pipeline2 | p2_base | real_correct | complete_pairwise | 0.377778 | 0.235657 | 0.071947 | 0.311111 | 0.311111 | 0.985763 | 0.340576 | 3.800000 | 5.700000 |
| pipeline2 | p2_base | real_mismatched | clean_unseen_compositions | 0.386806 | 0.440893 | 0.075717 | 0.018056 | 0.151864 | 1.000000 | 2.048479 | 8.625000 | 32.541667 |
| pipeline2 | p2_base | real_mismatched | complete_pairwise | 0.361111 | 0.239176 | 0.080076 | 0.294444 | 0.266667 | 0.988008 | 0.341695 | 3.822222 | 5.777778 |
| pipeline2 | p2_base | real_zero | clean_unseen_compositions | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| pipeline2 | p2_base | real_zero | complete_pairwise | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.166667 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### Synthetic comparable metrics by pairwise vs clean unseen-composition sets

Source: `artifacts/real_audio_distribution_shift/analysis_v1/synthetic_comparable_headline_metrics.csv`.

| pipeline | model_variant | condition | benchmark | target_event_hit_rate | pressed_key_f1 | timestep_f1 | ordered_event_accuracy | transition_event_pair_accuracy | max_unintended_key_state | integrated_unintended_key_state | wrong_press_count | unintended_press_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline1 | p1_base_basic_pitch | synthetic_basic_pitch | clean_unseen_compositions | 0.183333 | 0.381250 | 0.236133 | 0.183333 | 0.013889 | 0.978870 | 0.246073 | 0.875000 | 2.000000 |
| pipeline1 | p1_base_basic_pitch | synthetic_basic_pitch | complete_pairwise | 0.583333 | 0.750000 | 0.433626 | 0.583333 | 0.166667 | 0.345055 | 0.021817 | 0.166667 | 0.200000 |
| pipeline1 | p1_base_basic_pitch | synthetic_basic_pitch | exact_13_retention | 0.692308 | 0.756410 | 0.482253 | 0.692308 | 0.384615 | 0.289793 | 0.018001 | 0.153846 | 0.153846 |
| pipeline1 | p1_base_oracle | synthetic_oracle | clean_unseen_compositions | 0.256250 | 0.489286 | 0.346571 | 0.256250 | 0.114218 | 0.680594 | 0.121454 | 0.125000 | 0.125000 |
| pipeline1 | p1_base_oracle | synthetic_oracle | complete_pairwise | 0.583333 | 0.750000 | 0.433626 | 0.583333 | 0.166667 | 0.345055 | 0.021817 | 0.166667 | 0.200000 |
| pipeline1 | p1_base_oracle | synthetic_oracle | exact_13_retention | 0.692308 | 0.756410 | 0.482253 | 0.692308 | 0.384615 | 0.289793 | 0.018001 | 0.153846 | 0.153846 |
| pipeline2 | p2_base | synthetic_correct | clean_unseen_compositions | 0.339583 | 0.540675 | 0.268680 | 0.170833 | 0.013889 | 0.806667 | 0.259707 | 0.500000 | 1.625000 |
| pipeline2 | p2_base | synthetic_correct | complete_pairwise | 0.650000 | 0.794444 | 0.536169 | 0.650000 | 0.300000 | 0.322872 | 0.028108 | 0.133333 | 0.133333 |
| pipeline2 | p2_base | synthetic_correct | exact_13_retention | 0.769231 | 0.846154 | 0.671999 | 0.769231 | 0.538462 | 0.221166 | 0.015745 | 0.000000 | 0.000000 |

### Real-minus-synthetic distribution-shift deltas (calculated means from stored per-sequence delta rows)

| pipeline | model_variant | condition_real | benchmark | condition_synthetic | n | event_hit_delta_mean | pressed_key_f1_delta_mean | timestep_f1_delta_mean | max_unintended_delta_mean | integrated_unintended_delta_mean | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pipeline1 | p1_base_basic_pitch | real_basic_pitch | clean_unseen_compositions | synthetic_basic_pitch | 8 | 0.087500 | 0.126763 | 0.002769 | 0.021130 | 0.172773 | artifacts/real_audio_distribution_shift/analysis_v1/distribution_shift_effects.csv |
| pipeline1 | p1_base_basic_pitch | real_basic_pitch | complete_pairwise | synthetic_basic_pitch | 30 | 0.083333 | -0.098660 | 0.104897 | 0.654702 | 0.279167 | artifacts/real_audio_distribution_shift/analysis_v1/distribution_shift_effects.csv |
| pipeline1 | p1_base_oracle | real_oracle | clean_unseen_compositions | synthetic_oracle | 8 | 0.000000 | 0.000000 | 0.000000 | 0.000002 | 0.000011 | artifacts/real_audio_distribution_shift/analysis_v1/distribution_shift_effects.csv |
| pipeline1 | p1_base_oracle | real_oracle | complete_pairwise | synthetic_oracle | 30 | 0.100000 | -0.078889 | 0.227455 | 0.367449 | 0.103380 | artifacts/real_audio_distribution_shift/analysis_v1/distribution_shift_effects.csv |
| pipeline2 | p2_base | real_correct | clean_unseen_compositions | synthetic_correct | 8 | 0.061806 | -0.111341 | -0.207058 | 0.193333 | 1.979863 | artifacts/real_audio_distribution_shift/analysis_v1/distribution_shift_effects.csv |
| pipeline2 | p2_base | real_correct | complete_pairwise | synthetic_correct | 30 | -0.272222 | -0.558787 | -0.464222 | 0.662890 | 0.312468 | artifacts/real_audio_distribution_shift/analysis_v1/distribution_shift_effects.csv |

## 7. Data Availability / Missing Results

- Pipeline 2 seed61 long-horizon benchmark rows at lengths 3/5/10/20 were not found locally. Located seed61 outputs are Phase-A 13-sequence files under `artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/`.
- Transition-novelty vs timestep-F1 correlation could not be reported because no stored numeric transition-novelty column was found in the inspected result tables. Per-sequence archetype/category fields exist, but converting them into numeric novelty would be an extra definition rather than an unambiguous aggregation.
- Pair+Horizon and Pair-Only intermediate 100k/250k evaluation metrics were not found in the final analysis outputs. The local checkpoint files exist under `artifacts/paironly_hex/`, but evaluated metric CSVs located here report base and final adapted variants rather than all intermediate checkpoints.
- Basic Pitch pitch-presence precision and recall were not found as separate stored fields; only `pitch_presence_f1` plus strict expected/predicted/matched counts were found.
- Basic Pitch onset/offset MAE values in `artifacts/real_audio_distribution_shift/analysis_v1/basic_pitch_summary_interpretable.csv` are `NA` for the rows inspected; strict onset+offset F1 is stored as `strict_note_f1`.

## Source Inventory Notes

Primary source files used directly in this Appendix B evidence collation:

- `artifacts/five_note_factorial_1m_hex_export/analysis/per_condition_comparison.csv`
- `artifacts/five_note_factorial_1m_hex_export/analysis/best_checkpoint_per_condition.csv`
- `artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv`
- `artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/evaluation_checkpoint_metrics.csv`
- `artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/evaluation_checkpoint_metrics.csv`
- `artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed13_1m_retry1/evaluation/audio_dependence.csv`
- `artifacts/pipeline2_phase_a_hex/pipeline2_direct_audio_droq_v1_seed37_1m_retry1/evaluation/audio_dependence.csv`
- `artifacts/pipeline2_final_hex/pipeline2_direct_audio_droq_v1_seed61_1m/evaluation/audio_dependence.csv`
- `artifacts/paironly_final_evaluation/analysis_v1/long_horizon_by_length.csv`
- `artifacts/paironly_final_evaluation/analysis_v1/sequence_level_master.csv`
- `artifacts/paironly_final_evaluation/analysis_v1/headline_metrics.csv`
- `artifacts/paironly_final_evaluation/analysis_v1/pairwise_all25_summary.csv`
- `artifacts/paircomplete_refined_evaluation/analysis_v1/headline_metrics.csv`
- `artifacts/real_audio_distribution_shift/analysis_v1/headline_metrics.csv`
- `artifacts/real_audio_distribution_shift/analysis_v1/real_audio_headline_metrics.csv`
- `artifacts/real_audio_distribution_shift/analysis_v1/synthetic_comparable_headline_metrics.csv`
- `artifacts/real_audio_distribution_shift/analysis_v1/basic_pitch_summary_interpretable.csv`
- `artifacts/real_audio_distribution_shift/analysis_v1/distribution_shift_effects.csv`
