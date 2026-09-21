# TIDE × VBench-2.0 paired evaluation

A self-contained evaluation pipeline. It depends only on TIDE's own
`PromptRewriter`, DeepSeek configuration, and `SanaDiffusersT2VClient`, with no
imports from any sibling project.

By default it compares three strictly paired arms:

- `raw`: official canonical prompt, no skill;
- `initial`: a skill directory whose rules are the revision-0 initialization,
  passed via `INITIAL_SKILL`;
- `trained`: a trained frozen skill directory, passed via `TRAINED_SKILL`.

> Each arm is a **skill directory** containing both `SKILL.md` and `rules.json`
> (not a bare rules file). The original run used `wave_0000/frozen_skill` (rules
> revision 1) for `initial` and `wave_0011/frozen_skill` (rules revision 12) for
> `trained`, taken from its training output. Those `wave_*/frozen_skill`
> directories live under a training run's `outputs/` and are **not shipped** here.
> Set `INITIAL_SKILL` / `TRAINED_SKILL` to your own skill directories (for
> example the repository's `skills/rewrite-t2v-prompt`, or a copy of it whose
> `rules.json` is replaced with `initializations/posneg_v1_1.json`); the revision
> numbers here are provenance for the recorded run, not files to locate in this
> repository.

The same prompt uses the same seed across the three arms. Video filenames always
come from the canonical prompt, and evaluation metadata matches the official
`VBench2_full_info.json` exactly on the canonical prompt. The skill's conditional
negative prompt is merged (deduplicated) with the fixed negative prompt in the
SANA config; the final text is written into the manifest, job, and receipt for
auditing.

## Contents

- `prepare_skill_arms.py`: run the skill rewrite and prepare paired manifest/job;
- `generate_sana_shard.py`: one resident SANA model per GPU, supports resume;
- `audit_outputs.py`: check seed pairing, video hashes, receipts, and ffprobe info;
- `evaluate_vbench2.py`: call each official dimension's `compute_*` directly;
- `compare_results.py`: emit per-dimension scores and the delta vs. raw;
- `sample_prompt_suite.py`: re-fix a random sample from the official prompts;
- `data/`: reproducible Quick-44 and Full-18/random-3/seed-42 suites;
- `run_pipeline.sh`: unified entrypoint.

The legacy `scripts/prepare_vbench2_skill_ablation.py` and
`scripts/run_vbench2_skill_ablation.sh` are compatibility shims that execute the
implementation here.

No historical `outputs/`, Wan/VPO generation code, `old_vista/`, logs, videos, or
models were imported. The official VBench-2.0 checkout, model cache, and conda
environment remain external dependencies. Set `VBENCH2_ROOT` (and see
`model_env.sh`) to point at your local VBench-2.0 checkout and caches.

## No-side-effect validation first

This command only checks the suite, checkpoint, and task count; it does not call
DeepSeek or generate videos:

```bash
bash vbench2_eval/run_pipeline.sh validate
```

## Quick-44

Quick-44 covers 7 dimensions and 44 prompts. By default each ordinary prompt
generates 1 video; if the suite includes Diversity, that dimension automatically
uses at least 20 samples.

```bash
bash vbench2_eval/run_pipeline.sh prepare
bash vbench2_eval/run_pipeline.sh generate
bash vbench2_eval/run_pipeline.sh evaluate
```

You can also run all stages at once:

```bash
bash vbench2_eval/run_pipeline.sh all
```

Output defaults to `outputs/vbench2_skill_eval/`. Generation defaults to GPUs
0–5; the official VBench2 evaluation defaults to GPU 0 and runs per arm in
sequence to reduce memory conflicts.

## Full-18 / random-3 / seed-42

```bash
SUITE=vbench2_eval/data/vbench2_full18_random3_seed42.json \
OUTPUT_ROOT=outputs/vbench2_skill_eval_full18_r3_s42 \
SAMPLES_PER_PROMPT=3 \
bash vbench2_eval/run_pipeline.sh all
```

Full-18 has 54 canonical prompts. Diversity is raised to 20 samples even with
`SAMPLES_PER_PROMPT=3`; that is a minimum the code enforces.

## Common overrides

Smoke-test on a few prompts:

```bash
MAX_PROMPTS=2 OUTPUT_ROOT=outputs/vbench2_skill_eval_smoke \
bash vbench2_eval/run_pipeline.sh all
```

Select a subset of dimensions:

```bash
DIMENSIONS=Human_Anatomy,Composition \
OUTPUT_ROOT=outputs/vbench2_skill_eval_two_dims \
bash vbench2_eval/run_pipeline.sh all
```

Swap the trained checkpoint:

```bash
TRAINED_SKILL=/absolute/path/to/frozen_skill \
bash vbench2_eval/run_pipeline.sh all
```

`RESUME=1` is the default. A generated video is skipped when its receipt exists
and the video hash matches; when a video exists but the receipt is missing, a
resumed receipt is written. `audit_outputs.py` must still pass before a formal
comparison.

## Output structure

```text
outputs/vbench2_skill_eval/
├── experiment.json
├── rewrite_records.json
├── jobs/shard_*.jsonl
├── receipts/shard_*.jsonl
├── audit_summary.json
├── comparison_summary.json
├── raw/
│   ├── manifest.jsonl
│   ├── videos/<dimension>/*.mp4
│   ├── evaluation_metadata/*.json
│   ├── evaluation_results/*.json
│   └── evaluation_summary.json
├── initial/ ...
└── trained/ ...
```

The per-dimension paired delta in `comparison_summary.json` is the main
conclusion; the cross-dimension simple average is a diagnostic only, not an
official leaderboard total.

## Note on the trained checkpoint

The trained arm in the recorded run was the revision 12 skill saved at wave 11.
Earlier training records show that revision has uneven rule-module updates and
some module bloat, so it is suitable as a "this training endpoint" to validate,
not as a foregone conclusion that it beats the initial skill before the
per-dimension paired results are in. The `initial` arm is retained precisely to
separate "the benefit of rewriting itself" from "the gains/losses from training
updates." Whatever skill you pass via `TRAINED_SKILL`, keep the same `initial`
baseline to preserve that separation.
