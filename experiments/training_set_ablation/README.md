# Trigger-ablation VBench prompt rewrites

These two directories contain prompt rewrites produced with the final Skills
learned by the trigger ablation runs and `deepseek-v4-flash` through the
project's EpicLLMRouter configuration.

The selected VBench core subset is the same one used by the SANA ablation
plan: 72 `vbench_subject` prompts (subject consistency, motion smoothness,
dynamic degree) plus 93 `vbench_overall` prompts (overall consistency,
aesthetic quality, imaging quality). Each directory therefore contains 165
unique rewrites and 495 paired jobs using seeds 3407, 7859, and 12491.

- `no_trigger/`: final `ours_no_trigger_all_rules` Skill, activation mode
  `all` (all rules applied to every prompt).
- `random_trigger/`: final `ours_random_trigger` Skill, activation mode
  `random`, `random_activation_count=3`, seed `42`.

`prompts_165_unique.jsonl` stores one rewrite per source prompt;
`prompts_495.jsonl` and `jobs.jsonl` are expanded, generation-ready VBench
manifests. `rewrite_records.json` is the resumable API cache and
`experiment.json` records model, Skill/rules paths, hashes, dimensions, and
seeds.

The negative-prompt ablation is prepared in `negative_frozen/` after the
100-prompt SANA run at `outputs/ablation_negative_frozen` finishes. During
training, each rule's initial `negative_terms` is frozen. During Wan2.2
inference, rule-level conditional negatives are removed and only the shared
backend negative is used:

`chaotic sequence, deformed limbs, heavy motion blur, jump cuts, jerky movements, temporal artifacts, jitter, ghosting`

`run_negative_frozen_pipeline.sh` waits for training, expands the same 431
source prompts × 3 seeds (1293 jobs), generates Wan2.2 videos, audits them,
and runs the nine-dimension VBench evaluation.
