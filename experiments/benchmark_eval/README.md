# Benchmark generation workspace

This workspace decouples the benchmark manifest, prompt preparation, video
generation, and downstream evaluation. Seven arms are directly runnable,
**8,255 final videos** in total; outputs are saved per experiment directory and
evaluation is not started automatically.

For the overall evaluation plan see [`../EVALUATION_PLAN.md`](../EVALUATION_PLAN.md).

## Runnable arms

| arm | video model | prompt | VBench | CompBench | Total | Directory |
|---|---|---|---:|---:|---:|---|
| `sana_raw` | SANA-Video 2B | raw | 1,293 | 400 | 1,693 | `cross_model_results/sana-2b/` |
| `sana_ours` | SANA-Video 2B | trained Skill rev. 12 | 1,293 | 400 | 1,693 | `cross_model_results/sana-2b/` |
| `sana_ours_wo_neg` | SANA-Video 2B | trained positive only | 495 | 0 | 495 | `training_set_ablation/pos/` |
| `sana_init_skill` | SANA-Video 2B | initial `posneg_v1_1` | 495 | 0 | 495 | `training_set_ablation/init_skill/` |
| `cogvideox15_raw` | CogVideoX1.5-5B | raw | 1,293 | 0 | 1,293 | `cross_model_results/cogvideox1.5-5b/` |
| `cogvideox15_ours` | CogVideoX1.5-5B | trained Skill rev. 12 | 1,293 | 0 | 1,293 | `cross_model_results/cogvideox1.5-5b/` |
| `cogvideox15_vpo` | CogVideoX1.5-5B | VPO-5B | 1,293 | 0 | 1,293 | `cross_method_results/vpo/` |
| **Total** |  |  | **7,455** | **800** | **8,255** |  |

Unified generation parameters: `832×480`, `81 frames`, `16 fps`, `50 steps`,
guidance scale `6.0`. The three paired VBench seeds are fixed at
`3407 / 7859 / 12491`.

`sana_init_skill` reads
`skills/rewrite-t2v-prompt/initializations/posneg_v1_1.json` directly, so
`initializations/` is used by the **initial-Skill ablation rewrite** but not by
the trained `sana_ours` arm, which reads the trained
`working_skill/rules.json` produced by a training run.

> **On "trained Skill rev. 12".** The revision numbers in this table refer to the
> historical checkpoint used when the paper videos were recorded. The Skill
> shipped in this repository (`skills/rewrite-t2v-prompt/rules.json`) is a later
> revision, and `training_results/` keeps a frozen `v1.1-init` snapshot for
> reference. Deployment always reads the current `rules.json`; treat the "rev. 12"
> label as provenance for the recorded run, not as a file you must locate.

## Direct launch

Run from the project root. `all` runs the manifest check, prompt preparation,
sharded video generation, and SHA256 integrity audit in sequence, but does not
run evaluation:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 experiments/benchmark_eval/scripts/run_generation.sh sana_raw all
GPU_IDS=0,1,2,3,4,5,6,7 experiments/benchmark_eval/scripts/run_generation.sh sana_ours all
GPU_IDS=0,1,2,3,4,5,6,7 experiments/benchmark_eval/scripts/run_generation.sh sana_ours_wo_neg all
GPU_IDS=0,1,2,3,4,5,6,7 experiments/benchmark_eval/scripts/run_generation.sh sana_init_skill all
GPU_IDS=0,1,2,3,4,5,6,7 experiments/benchmark_eval/scripts/run_generation.sh cogvideox15_raw all
GPU_IDS=0,1,2,3,4,5,6,7 experiments/benchmark_eval/scripts/run_generation.sh cogvideox15_ours all
REWRITE_GPU=0 GPU_IDS=0,1,2,3,4,5,6,7 experiments/benchmark_eval/scripts/run_generation.sh cogvideox15_vpo all
```

Launch these one at a time based on available GPUs; each command occupies the
listed cards. The `ours` and initial-Skill prepare stages use the existing
DeepSeek router configuration. Each arm saves its own `rewrite_records.json`
snapshot; the underlying rewrite cache for identical Skill/rules is shared via
`benchmark_eval/cache/rewrites/`, so cross-model `ours` and `wo_neg` use the same
positive prompt. The VPO prepare stage loads a local VPO-5B on `REWRITE_GPU`.

You can also run the stages separately ("prepare all prompts first, then
generate"):

```bash
# Only check paths, weights, rules, and task count
experiments/benchmark_eval/scripts/run_generation.sh sana_ours validate

# Only build jobs.jsonl / experiment.json, no video generation
REWRITE_WORKERS=8 experiments/benchmark_eval/scripts/run_generation.sh sana_ours prepare

# Eight-GPU generation; rerunning the same command resumes by receipt + MP4 SHA256
GPU_IDS=0,1,2,3,4,5,6,7 experiments/benchmark_eval/scripts/run_generation.sh sana_ours generate

# Only verify all expected videos, seeds, prompt metadata, receipts, and file hashes
experiments/benchmark_eval/scripts/run_generation.sh sana_ours audit
```

A dry-run without loading models or writing fake MP4s is available before
launch:

```bash
experiments/benchmark_eval/scripts/run_generation.sh sana_raw prepare
DRY_RUN=1 MAX_TASKS=2 GPU_IDS=0 experiments/benchmark_eval/scripts/run_generation.sh sana_raw generate
```

Each shard's log goes to that arm's `logs/generate_shard_*`. By default a failed
job lets the rest of the shard continue and the run exits non-zero; after fixing
the problem the same command resumes.

## Per-arm output layout

Using `cross_model_results/sana-2b/sana_raw/` as an example:

```text
sana_raw/
├── experiment.json
├── jobs.jsonl
├── rewrite_records.json
├── logs/
├── receipts/                         # per-video provenance mirroring videos/
├── videos/
│   ├── main_vbench/
│   │   └── sana_raw/*.mp4            # must stay flat for VBench scanning
│   └── sana_compbench_diagnostic/
│       └── sana_raw/
│           ├── compbench_interaction/*.mp4
│           └── compbench_numeracy/*.mp4
└── generation_audit.json
```

A receipt records the raw/positive/conditional-negative/final-negative prompt,
seed, generation parameters, checkpoint, Skill revision/signature, GPU, elapsed
time, and video SHA256. Resume skips a video only when the receipt matches the
current task and MP4 hash exactly.

## Evaluation after generation

Start evaluation only after `generation_audit.json` reports `status: complete`.
For example, using an official VBench checkout you provide under `official/`:

```bash
python experiments/benchmark_eval/official/VBench/evaluate.py \
  --full_json_dir experiments/benchmark_eval/data/vbench/metadata/VBench_full_info.json \
  --videos_path experiments/cross_model_results/sana-2b/sana_raw/videos/main_vbench/sana_raw \
  --output_path experiments/benchmark_eval/outputs/eval_results/vbench/sana_raw \
  --dimension subject_consistency motion_smoothness dynamic_degree \
              aesthetic_quality imaging_quality overall_consistency \
              multiple_objects human_action spatial_relationship
```

Other arms only change `--videos_path` and `--output_path`. One batch of MP4s
supports several VBench dimensions; do not regenerate per dimension.

CompBench videos are placed under `compbench_interaction/` and
`compbench_numeracy/`. The official entrypoints are:

```text
official/T2V-CompBench/LLaVA/llava/eval/compbench_eval_interaction.py
official/T2V-CompBench/Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py
```

Before running the official eval you still need to prepare the LLaVA,
GroundingDINO/SAM environments and weights per the two official READMEs.

> `official/` is not shipped with this repository. Point it at your own VBench /
> T2V-CompBench checkouts (for example via symlinks) before evaluating.

## Currently blocked arms

| arm | reason |
|---|---|
| `sana_vista` | The current VistaAgent is bound to the TIDE training loop and lacks a standalone VISTA adapter that emits a single final video per benchmark seed. |
| `sana_ours_wo_update_limits` | No unrestricted-update training checkpoint. |
| `wan22_raw` / `wan22_ours` | Weights exist, but the official Wan2.2 inference checkout is missing; the existing Wan2.1 does not support `ti2v-5B`. |
| `wan22_video_prompting_skill` | The prompt Skill exists, but the batch executor is not fixed and the Wan2.2 runtime is likewise missing. |

These arms are registered in `config/generation_registry.json`; the unified
launcher refuses to run them rather than silently swapping models or inference
implementations.

## Manifest source

```bash
python experiments/benchmark_eval/scripts/validate_bundle.py
python experiments/benchmark_eval/scripts/build_manifests.py
```

The three source manifests total 14,322 entries covering ready and blocked arms.
The generation registry filters out the 8,255 entries that currently run
reliably; evaluation always targets the original benchmark prompts.
