# Self-improvement contract

This contract turns VISTA-style test-time feedback into reusable rewrite behavior. It
separates expensive **training/search** from cheap **deployment**:

```text
training: raw baseline + skill rewrite -> T2V -> two rounds of critique
          -> four optimized prompt pairs -> binary tournament
          -> one-rule candidate patches -> persistent pool
          -> four same-rule patches -> one atomic single-rule update
deployment: prompt -> frozen skill rewrite -> one T2V video
```

The video is the evaluator during training, not a required input at deployment. Treat the
same-rule four-patch batch like a mini-batch gradient: repeated selected behavior is positive
evidence, repeated losing behavior is negative evidence, and conflicting evidence should
narrow or retain the instruction. Preserve the trigger unless repeated evidence identifies
a routing error rather than an instruction-quality error.

## v1.1 trigger contract

Treat each module's `trigger_axis` as a frozen routing responsibility and its `trigger`
as a conservatively trainable decision boundary. Distinct conditional rules must have
distinct axes; global rules are listed in `trigger_policy.global_rule_ids`. Multiple
rules may activate for one source only when independent source facts satisfy their
separate axes. Do not infer one axis from another activated rule.

Preserve a trigger by default. A trajectory may propose a minimal edit only when it
directly shows a missed activation or an unsupported activation. A four-patch batch may
apply that edit only when at least two patches independently support compatible routing
changes. The learned trigger must remain within its existing axis, stay within the length
limit, and remain below the configured lexical-overlap threshold against every other
conditional trigger. `trigger_axis` and `capability_scope` remain immutable. Adding,
deleting, splitting, merging, or changing an axis is an explicit Skill-version migration.

## Per-prompt trajectory schema

```json
{
  "schema_version": "1.0",
  "source_prompt": "...",
  "initial_skill_rewrite": {"positive_prompt": "...", "negative_prompt": "..."},
  "raw_baseline_video": {},
  "initial_rewrite_video": {},
  "optimization_rounds": [
    {"round_index": 1, "critiques": [], "candidate_prompts": [], "tournament": {}},
    {"round_index": 2, "critiques": [], "candidate_prompts": [], "tournament": {}}
  ],
  "final_champion": {}
}
```

The raw video is a diagnostic baseline. The first critique starts from the frozen-skill
rewrite, not from a raw-vs-rewrite tournament. Each optimization round emits exactly four
new positive/negative prompt pairs and uses a binary tournament to choose its champion.

## Rule mini-batch update schema

```json
{
  "schema_version": "1.0",
  "status": "update",
  "reasoning": "batch-level account of gains, losses, conflicts, and safeguards",
  "evidence_episode_ids": ["four episode ids"],
  "module": {"rule_id": "selected existing rule", "trigger": "preserved or minimally learned trigger",
             "trigger_axis": "exact current trigger axis",
             "instruction": "...", "negative_terms": [],
             "capability_scope": [], "expected_side_effects": []}
}
```

## Atomic update

Freeze the rule-initialized Skill throughout a Wave. Store every unused candidate in the
cross-Wave pool. Select at most one rule from the current Wave, supplement its patches from
the same-rule pool, and do nothing when fewer than four exist. When four exist, perform one
text-only distillation call, validate that only the selected rule changed, consume those
four patches, increment the revision once, and atomically replace `rules.json`. Keep the
frozen snapshot, pool selection, reasoning, and revisions for audit and rollback.

## What the patch learner must not do

- Do not update from one video or one trajectory.
- Do not mention VBench, a judge, a benchmark category, or a particular seed in a rule.
- Do not add generic style words (“more cinematic”, “higher quality”) without a concrete
  visual failure and trigger.
- Do not treat verbosity as improvement or let one case erase unrelated behavior.
- Do not add a rule or change `trigger_axis` or `capability_scope` during trajectory
  compression, mini-batch distillation, or promotion.
- Do not change a trigger from downstream quality evidence alone, from only one batch
  patch, across another rule's axis, or when the orthogonality audit fails.
