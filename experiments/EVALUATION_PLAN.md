# TIDE evaluation plan

Updated: 2026-08-10

## 1. Positioning

Main results center on **VBench**, which has broader recognition and coverage;
T2V-CompBench only supplements the physical-interaction and exact-counting
abilities that VBench lacks. All methods use the same raw prompts, the same video
generation parameters, and paired random seeds, and the evaluator always targets
the original benchmark prompt.

This plan generates 3 videos per prompt in VBench. That is below the official
5-sample setting, so the paper should describe it as a
`VBench subset evaluation with three fixed random seeds per prompt` and not call
the numbers the official VBench Total Score or compare them directly with the
official leaderboard.

## 2. Experiment groups

Not every arm below is directly runnable in this release. Arms marked
*(blocked)* need an external runtime or a checkpoint that is not shipped; see the
"Currently blocked arms" table in
[`benchmark_eval/README.md`](benchmark_eval/README.md) for the reason and the
registry that refuses to launch them.

### SANA-Video 2B

- `sana_raw`: raw prompt
- `sana_vista`: raw prompt + VISTA *(blocked: no standalone VISTA benchmark adapter)*
- `sana_ours`: raw prompt + TIDE
- `sana_ours_wo_neg`: drop the learned negative prompt
- `sana_ours_wo_update_limits`: drop the skill-update limits *(blocked: no unrestricted-update checkpoint)*
- `sana_init_skill`: initial skill only

### CogVideoX1.5-5B

- `cogvideox15_raw`
- `cogvideox15_vpo`
- `cogvideox15_ours`

### Wan2.2-TI2V-5B

- `wan22_raw` *(blocked: missing official Wan2.2 inference checkout)*
- `wan22_video_prompting_skill` *(blocked: missing Wan2.2 runtime / fixed batch executor)*
- `wan22_ours` *(blocked: missing official Wan2.2 inference checkout)*

## 3. VBench main evaluation

The 9 main experiment groups are all evaluated on these 9 dimensions:

- Quality/degradation guards: `subject_consistency`, `motion_smoothness`,
  `dynamic_degree`, `aesthetic_quality`, `imaging_quality`;
- Semantics/controllability: `overall_consistency`, `multiple_objects`,
  `human_action`, `spatial_relationship`.

With shared prompts across dimensions, each group needs only:

```text
(72 subject + 93 overall + 82 multiple_objects
 + 100 human_action + 84 spatial_relationship) × 3 seeds
= 431 × 3
= 1,293 videos
```

The 9 main groups total `9 × 1,293 = 11,637` final evaluation videos.

The three SANA ablations first use the shared core 6 dimensions:

```text
subject_consistency / motion_smoothness / dynamic_degree
overall_consistency / aesthetic_quality / imaging_quality
```

Each ablation is `(72 + 93) × 3 = 495` videos; the three ablations total 1,485
videos. If the trimmed ablation shows a clear difference, additionally run
`multiple_objects`, `human_action`, and `spatial_relationship`.

## 4. T2V-CompBench supplement

Keep only the two dimensions with no direct VBench counterpart:

- `object_interactions`: contact, collision, support, physical response;
- `generative_numeracy`: exact count, repetition, disappearance, count drift.

Run these only on `sana_raw`, `sana_vista`, and `sana_ours` by default. Each
group is `2 × 200 = 400` videos; the three groups total 1,200 videos.

## 5. Current budget

| Part | Videos |
|---|---:|
| VBench 9 dims across 9 main experiments | 11,637 |
| VBench core 6 dims across 3 SANA ablations | 1,485 |
| CompBench 2 dims across 3 SANA main methods | 1,200 |
| **Total** | **14,322** |

This counts videos fed into the benchmark. Methods like VISTA and VPO that
internally generate multiple candidates before selecting a final video need
their actual T2V inference call count recorded separately; 14,322 does not stand
in for it.

## 6. Reproducibility requirements

- The three VBench seeds are fixed at `3407`, `7859`, `12491`; strictly paired
  across methods.
- Save the raw prompt, positive prompt, negative prompt, seed, generation
  parameters, model version, skill revision, `rules.json` SHA256, and final video
  path.
- After using the prompt enhancer, still name videos by the original benchmark
  prompt or original index.
- Keep every CompBench index; do not deduplicate on repeated text.
- Run `benchmark_eval/scripts/validate_bundle.py` first, then
  `benchmark_eval/scripts/build_manifests.py` to build the per-video task list.

## 7. Completed supplement: VISTA self-improve matched-subset triplet (2026-08-23)

To validate `vista_self_improve` first, an 80-prompt subset from
`experiments/cross_method_results/vista/vista_vbench_prompts.txt` was used for a
three-group matched VBench2 evaluation. This subset covers 5 VBench dimensions,
16 prompts each:

- `human_action`: 16;
- `subject_consistency` / `motion_smoothness` / `dynamic_degree`: 16;
- `multiple_objects`: 16;
- `overall_consistency` / `aesthetic_quality` / `imaging_quality`: 16;
- `spatial_relationship`: 16.

The three groups are:

- `vista_self_improve`: the `best_video` recorded in the VISTA self-improvement
  trajectory per prompt;
- `sana_ours`: SANA-Video 2B + TIDE video for the same raw prompt, taking
  `-0.mp4` (seed 3407);
- `sana_raw`: SANA-Video 2B raw video for the same raw prompt, taking `-0.mp4`
  (seed 3407).

Each group has 80 videos, 240 final evaluation videos total, evaluated on all 9
main VBench2 dimensions in standard mode. Because VISTA currently has only one
final selected video per prompt, the two SANA groups also take one video per
prompt for a matched sample count. This experiment reuses existing videos and
adds no new generation budget.

### 7.1 Results

VBench2 aggregate values:

| Dimension | `vista_self_improve` | `sana_ours` | `sana_raw` |
|---|---:|---:|---:|
| `subject_consistency` | 0.967677 | 0.958119 | 0.960664 |
| `motion_smoothness` | 0.988769 | 0.987124 | 0.985508 |
| `dynamic_degree` | 0.625000 | 0.687500 | 0.875000 |
| `aesthetic_quality` | 0.612125 | 0.612994 | 0.612500 |
| `imaging_quality` | 0.701585 | 0.678012 | 0.677877 |
| `overall_consistency` | 0.276320 | 0.282555 | 0.289679 |
| `multiple_objects` | 0.945312 | 0.898438 | 0.789062 |
| `human_action` | 1.000000 | 1.000000 | 0.812500 |
| `spatial_relationship` | 0.872375 | 0.886982 | 0.815909 |

The unweighted mean over the nine dimensions is a convenience summary for this
subset only:

```text
vista_self_improve: 0.776574
sana_ours:          0.776858
sana_raw:           0.757633
```

Relative to `sana_raw`, VISTA is higher on `multiple_objects`, `human_action`,
`spatial_relationship`, `subject_consistency`, and `imaging_quality`, but clearly
lower on `dynamic_degree`; versus `sana_ours` the overall subset mean is roughly
even.

### 7.2 Results and reproduction paths

> The directories below are the output locations of the original run and are
> **not shipped** with the code release (generated videos, per-video evaluation
> results, and logs are git-ignored). They are listed to document the original
> layout; regenerate the equivalent outputs with the `benchmark_eval` pipeline
> (`run_generation.sh` + the official VBench evaluation) using the pinned prompt
> list `experiments/cross_method_results/vista/vista_vbench_prompts.txt`.

- Matched video list:
  `experiments/cross_method_results/vista/vista_vbench_eval/matched_triplet_1sample/`;
- GPU 0/1 rerun results:
  `experiments/cross_method_results/vista/vista_vbench_eval/triplet_results_gpu01/`;
- Per-group logs:
  `experiments/cross_method_results/vista/vista_vbench_eval/triplet_logs_gpu01/`;
- Summary notes:
  `experiments/cross_method_results/vista/vista_vbench_eval/triplet_results_gpu01/README.md`.

This is an 80-prompt matched-subset evaluation, not the official full VBench
Total Score; it should not be compared directly with VBench leaderboard values,
and the nine-dimension unweighted mean is not a formal paper total.
