# TIDE method: patch extraction and Skill update pipeline

> Implementation-checked description of the mechanism the code actually runs:
> from revision-0 Skill initialization, per-prompt trajectory generation, and
> per-rule patch extraction, through per-rule FIFO scheduling, atomic Skill
> updates, crash recovery, and the end-of-training `final_flush`.
>
> Scope: this documents the *code* mechanism, not a specific trained checkpoint.
> Revision numbers below (e.g. revision-0, `no_op_consumed`) describe the update
> protocol; the concrete revision shipped in
> `skills/rewrite-t2v-prompt/rules.json` will differ.

## 1. Core goals

The pipeline solves two problems:

1. One video prompt may trigger several rules (motion, identity, scene, ...) at
   once, so the learning signal must no longer be assigned to only one rule.
2. High-frequency rules tend to monopolize update opportunities, so patches must
   be queued per rule and rules that have waited longer (already past the update
   threshold) should update first.

Core constraints:

- Assess exactly as many rules as the rewrite activated.
- A trigger hit only means "gets a chance to be assessed", not "must be modified".
- One trajectory produces at most one candidate patch per rule.
- A normal update merges exactly 4 patches of the same rule.
- Each wave updates at most one rule.
- Strict FIFO within a rule.
- Consume patches only after the Skill commit succeeds.
- After all prompts finish, keep processing full batches of 4, then use
  `final_flush` to process each rule's remaining 1–3 patches.
- When the content does not actually change, record `no_op_consumed`: consume the
  evidence but do not bump the revision.

## 2. End-to-end flow

```text
revision-0 rules
        │
        ▼
create run-local working_skill
        │
        ▼
freeze the current Skill at each Wave ───┐
        │                                │
        ▼                                │
per-prompt Skill rewrite                 │
emit positive / negative / activated IDs │
        │                                │
        ▼                                │
SANA generates video + Qwen critique /   │
tournament; DeepSeek writes optimized    │
prompts, forming a complete trajectory   │
        │                                │
        ▼                                │
DeepSeek assesses all activated rules:   │
candidate / no_change / misrouted / reject
        │
        ▼
only candidates enter that rule's FIFO queue
        │
        ▼
any rule with queue length >= 4?
        ├── no: this Wave is no_update
        └── yes: pick the eligible rule that has waited longest
                     │
                     ▼
              take the earliest 4 patches from its queue
                     │
                     ▼
              DeepSeek merges a full rule replacement
                     │
                     ▼
              schema / trigger / evidence / length validation
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
      atomic commit ok    merge or validation failed
           │                   │
           ▼                   ▼
      consume 4 patches    queue unchanged
           │
           └──────────────► next Wave

all prompts finished
        │
        ▼
finalization: first drain all 4-patch batches
        │
        ▼
merge each rule's remaining 1–3 patches
        │
        ▼
all queues empty, finalization_complete = true
```

## 3. Revision-0 initialization and Wave isolation

### 3.1 Initialization source

Training starts from:

```text
skills/rewrite-t2v-prompt/initializations/posneg_v1_1.json
```

The controller requires the initialization file's `revision` to be 0. When
starting a brand-new output directory, it:

1. Copies the full project `skills/rewrite-t2v-prompt` into
   `<output_root>/working_skill`;
2. Overwrites `working_skill/rules.json` with the specified revision-0
   initialization;
3. Saves the initialization file path, SHA256, initial Skill SHA256, and
   initialization time;
4. Re-checks that the run-local Skill's revision is really 0.

The record file is:

```text
<output_root>/skill_initialization.json
```

Training only modifies:

```text
<output_root>/working_skill/rules.json
```

It never directly overwrites the project's top-level deployed Skill. On resume,
the initialization file's SHA256 must match the value from when the run was
created.

### 3.2 Frozen Skill

At the start of each Wave, the controller copies the current `working_skill` to:

```text
wave_XXXX/frozen_skill/
```

All prompts in that Wave use the same frozen revision. Only after every prompt in
the Wave finishes and patch collection completes may the controller update
`working_skill`, so the next Wave sees the new revision.

## 4. How the rewrite stage records rule hits

### 4.1 Output contract

`PromptRewriter` requires DeepSeek to return:

```json
{
  "positive_prompt": "rewritten T2V prompt",
  "negative_prompt": "conditional artifact terms",
  "activated_rule_ids": [
    "vista.intent_preservation",
    "vista.motion_physics"
  ]
}
```

So one rewrite may trigger several rules at once, not just one.

### 4.2 Activation normalization

After the model responds, the code:

1. Drops rule IDs that do not exist in `rules.json`;
2. Removes duplicate IDs, keeping first-seen order;
3. Prepends `trigger_policy.global_rule_ids` to the activation list;
4. In v1.1, `vista.intent_preservation` is a global rule, so every prompt checks
   it;
5. Other conditional rules should only be hit by the source prompt's own trigger,
   never cascaded because another rule already fired.

If the model's positive prompt is too short, the system falls back to the
source/user prompt but keeps the normalized activation sidecar.

### 4.3 Negative-prompt restriction

The rewrite's negative prompt cannot be invented freely. The system only allows
terms from the `negative_terms` of the modules named in this rewrite's
`activated_rule_ids`:

- negative terms from rules that were not hit are removed;
- new terms not in the hit modules' vocabulary are removed;
- after deduplication, at most 8 remain.

This makes the positive instruction, trigger activation, and negative prompt use
the same rule-routing result.

### 4.4 Thinking configuration

- prompt rewrite: DeepSeek, `thinking=False`;
- downstream prompt optimization: DeepSeek, `thinking=False`;
- video critique and tournament: Qwen, thinking always off;
- per-rule patch assessment: DeepSeek, `thinking=True`;
- merging 1–4 patches into a Skill: DeepSeek, `thinking=True`;
- schema repair when a batch output is invalid: DeepSeek, `thinking=False`.

## 5. How a single prompt forms a learning trajectory

The standard configuration uses 10 videos per prompt:

```text
Round 0: 1 raw prompt + 1 frozen-Skill rewrite
Round 1: 4 optimized prompts, 1 video each
Round 2: 4 optimized prompts, 1 video each
Total: 1 + 1 + 4 + 4 = 10
```

Execution order:

1. Save the raw baseline;
2. Do a one-pass rewrite with the frozen Skill and save `activated_rule_ids`;
3. Use the Skill rewrite result as the initial incumbent;
4. Qwen does video understanding and multi-dimension critique of the current
   champion;
5. DeepSeek generates 4 improved prompts from the source prompt, champion, and
   critique;
6. SANA-Video generates a video for each;
7. Qwen tournament picks the winner;
8. Repeat one more optimization round;
9. Save the final champion and compress the full search into a
   `skill_learning_episode`.

The complete `trajectory.json` contains the source prompt, initial rewrite,
activation, per-round critiques, candidate positive/negative prompts, videos,
tournament results, and final champion. This trajectory evidence is then used to
judge whether each activated rule actually needs to change.

## 6. Extracting a patch for every hit rule

### 6.1 One call, many assessments

After the trajectory completes,
`SkillDistiller.propose_trajectory_rule_assessments()` puts the full
`skill_learning_episode`, all `activated_rule_ids`, and the corresponding full
modules into a single DeepSeek thinking call. The model must return one
assessment per activated rule, in activation order. One trajectory can therefore
produce independent candidate patches for motion, identity, scene, etc.

If the model omits an activated rule, the code auto-fills a `reject` for it
(reason: distiller omitted assessment); output for unknown, duplicate, or
non-activated rules is not accepted.

### 6.2 Four assessment states

| State | Meaning | Enters patch queue? |
|---|---|---|
| `candidate` | The trajectory shows a reusable new defect/improvement within this rule's responsibility | Yes |
| `no_change` | The trigger hit is reasonable, but no evidence that the rule needs to change | No |
| `misrouted` | Direct evidence the rule was a false-positive activation for the source prompt | Becomes a candidate after passing trigger validation |
| `reject` | Evidence missing, conflicting, out of scope, malformed, or unattributable | No |

The key distinction:

```text
trigger hit != automatic patch
```

### 6.3 Conditions for a candidate

The model is asked to return a candidate only when:

- The failure is within the rule's `trigger_axis` and `capability_scope`;
- The critiques, candidate prompts, tournament, and final champion provide
  reusable evidence;
- The proposed instruction adds or corrects something substantive relative to the
  current instruction;
- The content can be abstracted into general rewrite behavior, not memorizing a
  specific prompt, entity, video, benchmark, model, seed, or path;
- The edit is a full replacement proposal for the current rule, not a cross-rule
  partial diff.

### 6.4 Special handling of `misrouted`

`misrouted` is only for a direct trigger false positive, not ordinary
video-quality failures. The code normalizes a valid `misrouted` into a
trigger-only candidate:

- the instruction copies the current rule, no change allowed;
- negative terms copy the current rule;
- expected side effects copy the current rule;
- only a minimal trigger narrowing is allowed;
- a `trigger_change_rationale` is required;
- the trigger must pass the boundary and orthogonality checks.

If the proposed trigger does not actually change, lacks a rationale, or fails
trigger validation, the assessment becomes a `reject` and does not enter the
queue.

### 6.5 Normalized patch

The key structure of a valid candidate:

```json
{
  "schema_version": "3.0",
  "patch_id": "<episode_id>:<rule_id>",
  "episode_id": "...",
  "rule_id": "vista.motion_physics",
  "status": "candidate",
  "assessment_status": "candidate",
  "reasoning": "...",
  "confidence": 0.8,
  "base_rule_hash": "...",
  "edit": {
    "operation": "replace",
    "rule_id": "vista.motion_physics",
    "trigger": "...",
    "trigger_axis": "motion.kinematics_physics",
    "instruction": "complete reusable replacement",
    "negative_terms": ["..."],
    "capability_scope": ["..."],
    "expected_side_effects": ["..."],
    "trigger_change_rationale": ""
  }
}
```

`patch_id = episode_id:rule_id`, so one episode can produce one patch per
different rule, but not multiple patches for the same rule. `base_rule_hash` is
recorded for auditing; the current controller does not yet auto-evict stale
patches that disagree with a new revision.

## 7. Persistence and recovery of single-prompt results

Each completed prompt writes the following into both `trajectory.json` and its
item `status.json`:

- `skill_learning_episode`;
- `activated_rule_ids`;
- `rule_assessments`;
- `candidate_patches`;
- best prompt, best video, and trajectory path.

If the 10 videos and final champion are done but the final DeepSeek
rule-assessment call fails, recovery reads the existing `trajectory.json` and
retries only the assessment, without regenerating videos. The resume condition
is:

- the source prompt matches the current item;
- a valid `best_video` and prompt candidate already exist;
- `rule_assessments` has not been written yet.

## 8. Patch collection and per-rule FIFO pool

### 8.1 Wave collection

After all shards in a Wave finish, the controller scans:

```text
wave_XXXX/shard_*/items/*/status.json
```

Only records satisfying all of the following become enqueueable patches:

- item status is `completed`;
- a `learning_episode` exists;
- the patch status is `candidate`;
- `edit.rule_id` is non-empty.

All assessments feed the Wave's classification statistics, but only candidates
enter the active pool.

### 8.2 Pool data structure

The body of `<output_root>/patch_pool.json`:

```json
{
  "schema_version": "3.0",
  "queues": {
    "vista.motion_physics": [],
    "vista.identity_count_guard": [],
    "vista.scene_components": []
  },
  "rule_state": {
    "vista.motion_physics": {
      "eligible_since_wave": null,
      "last_updated_wave": null,
      "last_processed_wave": null
    }
  },
  "consumed_patch_ids": [],
  "consumed_batches": []
}
```

Each new patch is appended to its rule's queue tail and records `enqueued_wave`
and `enqueue_index`. The system deduplicates pending patch IDs against
`consumed_patch_ids`, so already-used patches are not re-enqueued on resume or
rescan.

## 9. Normal-update FIFO and wait-priority scheduling

### 9.1 Eligibility

The normal batch size is fixed at 4. When a rule's queue length first reaches 4,
it records:

```text
eligible_since_wave = current Wave
```

While the queue is below 4 this field is `null`.

### 9.2 Actual scheduling order

Each Wave selects at most one eligible rule. The current sort key is, in order:

1. earlier `eligible_since_wave` (waited longer after reaching 4);
2. earlier `last_updated_wave` (never-updated rules treated as `-1`);
3. longer current queue;
4. `rule_id` lexicographic order, for fully deterministic final tie-breaking.

The current implementation does not use patch confidence in the scheduler
tie-break. Low-frequency rules whose queue never reaches 4 are not forcibly
modified just because they have not updated in a long time.

### 9.3 Strict FIFO patch selection

The selected rule reads only the earliest 4 in its queue:

```python
chosen = pool["queues"][selected_rule][:4]
```

Patches from different rules are never mixed to fill a batch; a later patch
cannot jump ahead of an older patch of the same rule.

## 10. Merging four patches into one rule update

### 10.1 Merge input

The DeepSeek thinking call receives only:

- the current rule;
- 4 candidate patches pointing at the same `rule_id`;
- the instruction soft-warning threshold (not a hard cap);
- update type `normal_batch_update`.

It must output one complete module replacement and cite all 4 episode IDs. The
merge goal is to distill repeated evidence, resolve conflicts, and compress into
general principles — never a naive concatenation of the 4 patches.

### 10.2 Mandatory validation

The model result must pass:

- `status == update`;
- only the selected `rule_id` returned;
- non-empty instruction;
- no hard character cap on the instruction; `update_policy.instruction_warning_chars`
  (8,000 in v1.1) is only a length-audit warning threshold and exceeding it is
  not rejected;
- the instruction must not contain the code's explicitly blocked evaluator-leak
  words;
- `evidence_episode_ids` must exactly match the selected patches' episode ID set;
- `rule_id` unchanged;
- `trigger_axis` force-copied from the old value;
- `capability_scope` force-copied from the old value;
- negative terms emptied of blanks and truncated to at most 8;
- all other modules completely unchanged.

If the first result is invalid, the system allows one `thinking=False`
serialization/schema repair. Repair can only restore original answer fields, not
create new substantive updates; exceeding the warning threshold is not itself
illegal — only an empty instruction, structural error, or evaluator leak causes a
second invalid result and raises an error, leaving the patch queue unchanged.

### 10.3 Trigger-update gate

The trigger stays unchanged by default. It is modified only when all of the
following hold:

1. At least 2 selected patches independently proposed a trigger change with a
   routing rationale;
2. The batch merge result also proposes a trigger change;
3. `trigger_policy.trigger_trainable == true`;
4. The trigger does not exceed the max length;
5. A conditional trigger cannot become an unbounded global trigger;
6. The word-set Jaccard against other conditional triggers is below the threshold
   (currently 0.35);
7. The change does not cross the frozen `trigger_axis`.

When evidence is insufficient, conflicting, or fails the orthogonality check, the
system keeps the old trigger and records `discarded_trigger_change`, the support
count, and validation details.

### 10.4 `update` vs. `no_op_consumed`

The normalized module is compared with the current module:

- any actual field change: `status = update`, revision += 1;
- identical: `status = no_op_consumed`, revision unchanged.

Both states consume the checked patches, preventing the same evidence from
re-entering training.

## 11. Atomic commit, consumption order, and crash recovery

### 11.1 Commit order

The normal success path:

```text
peek 4 patches
    ↓
DeepSeek merges and passes validation
    ↓
verify base_revision
    ↓
write full rules.json to a temp file
    ↓
os.replace atomically swaps working_skill/rules.json
    ↓
write skill_update.json
    ↓
consume patches from the active queue
    ↓
update consumed_patch_ids / consumed_batches
```

Only `status=update` actually rewrites `rules.json`; `no_op_consumed` does not
write a new revision but still persists the result and consumes patches. If merge
or validation raises, the consumption step does not run.

### 11.2 Post-commit crash

The Skill update may already be atomically committed, but the process may exit
before writing `skill_update.json` or consuming the pool. On recovery, the
controller compares the Skill's

```text
last_batch_update.evidence_episode_ids
```

with this batch's selected episode IDs. If they match exactly, it treats the
Skill as committed, produces a `recovered_after_commit` result, only backfills
the update record and patch consumption, and does not bump the revision again.

If `skill_update.json` already exists, recovery idempotently redoes consumption;
`consumed_patch_ids` and `consumed_batches` both have dedup protection.

## 12. `final_flush` after all prompts finish

### 12.1 When it starts

Only when

```text
train_offset == total prompt count
```

does the controller enter `finalize_patch_pool()`. If training prompts are not
finished, the run is not treated as completed.

### 12.2 Two-phase drain

The finalization loop:

1. While any rule queue has length >= 4, keep applying the normal
   eligible-wait/FIFO rules, processing the earliest 4 each time, type still
   `normal_batch_update`;
2. Once all queues are below 4, select a rule with 1–3 remaining patches and merge
   its remaining patches at once, type `final_tail_update`;
3. Repeat until all queues are empty.

So a rule's remaining 3 patches merge into one update, not 3 revisions;
different rules' tail patches are not mixed.

### 12.3 Tail rule order

The order for the 1–3-patch tail queues is, in order:

1. earlier `last_updated_wave`, never-updated rules first;
2. earlier `enqueued_wave` of the earliest patch in the queue;
3. `rule_id` lexicographic order.

### 12.4 Conservative tail constraints

The tail batch still uses the same full replacement, length audit, evidence,
trigger, and atomic commit validation, but the prompt explicitly requires: with
fewer than 4 pieces of evidence, only make the narrow changes those patches
directly support.

A trigger change still requires at least 2 supporting routing patches, so:

- 1 tail patch cannot modify a trigger;
- 2–3 tail patches can modify a trigger only with at least 2 compatible routing
  evidences.

A tail result identical to the current rule likewise records `no_op_consumed`,
clears the patches, and does not bump the revision. If merge or validation fails,
the patches are not consumed and finalization is not wrongly marked complete.

### 12.5 Completion condition

Only when all rule queues in the pool are empty does
`final_updates/finalization_state.json` write:

```json
{
  "completed": true,
  "remaining_queue_lengths": {
    "vista.intent_preservation": 0,
    "vista.motion_physics": 0
  }
}
```

Then `controller_state.json` sets `finalization_complete = true`. The supervisor
checks both the training prompt count and this field, avoiding a false
"completed" when 100 prompts finished but tail patches are unprocessed.

## 13. Output directory and the role of each file

```text
<output_root>/
├── skill_initialization.json
│   └── revision-0 source, SHA256, and run-local Skill info
├── controller_state.json
│   └── next_wave, train_offset, revision, finalization_complete
├── controller_summary.json
│   └── controller current/final summary and remaining_prompts
├── patch_pool.json
│   └── per-rule pending FIFO, rule_state, and the consumed ledger
├── working_skill/
│   └── rules.json                       # current trained Skill
├── wave_XXXX/
│   ├── frozen_skill/rules.json          # revision actually used by this Wave
│   ├── batch_manifest.json              # Wave config, timing, status, update summary
│   ├── rule_assessment_summary.json     # per-rule four-class assessment counts and enqueued IDs
│   ├── selected_rule_batch.json         # selected rule and FIFO patches when there is an update
│   ├── skill_update.json                # update / no_op / no_update result
│   ├── batch_distillation/
│   │   ├── attempt_01.json              # thinking merge raw response and validation
│   │   └── attempt_02_repair.json       # only appears if the first was invalid
│   └── shard_*/items/*/
│       ├── status.json                  # activation, assessments, patches, best result
│       └── run*/trajectory.json         # full video-optimization and learning trajectory
└── final_updates/
    ├── finalization_state.json          # resumable final-flush progress
    └── update_XXXX/
        ├── selected_rule_batch.json
        ├── skill_update.json
        └── batch_distillation/
```

To use the Skill after a given Wave, read `working_skill/rules.json` after that
Wave; to exactly reproduce a Wave's rewrite input, use that Wave's own
`frozen_skill/`.

## 14. Current-implementation pseudocode

```python
working_skill = initialize_from_revision_0()

for wave in waves:
    frozen_skill = freeze(working_skill)

    for prompt in wave:
        rewrite = rewrite_with_skill(prompt, frozen_skill)
        activated_ids = known_ids(global_ids + rewrite.activated_rule_ids)
        trajectory = run_two_round_video_search(rewrite)

        assessments = assess_all_activated_rules(
            trajectory,
            activated_ids,
            thinking=True,
        )

        for assessment in assessments:
            persist_for_audit(assessment)
            if assessment.status == "candidate":
                fifo[assessment.rule_id].append_unique(assessment)

    eligible = [rule for rule in fifo if len(fifo[rule]) >= 4]
    if eligible:
        rule = select_by_eligible_since_then_last_update_then_queue_length(eligible)
        patches = fifo[rule][:4]
        result = merge_validate_and_commit(rule, patches)
        if result.status in {"update", "no_op_consumed"}:
            consume_after_commit(patches)

if all_prompts_finished:
    while any(len(queue) >= 4 for queue in fifo.values()):
        process_one_normal_fifo_batch()

    while any(1 <= len(queue) <= 3 for queue in fifo.values()):
        rule = select_oldest_tail_rule()
        patches = list(fifo[rule])
        result = merge_validate_and_commit(rule, patches, conservative=True)
        if result.status in {"update", "no_op_consumed"}:
            consume_after_commit(patches)

    finalization_complete = all(len(queue) == 0 for queue in fifo.values())
```

## 15. What the current mechanism solves, and what to still watch

The current design already provides:

- visible multi-rule activation;
- an independent assessment for every hit rule;
- clear semantics for `no_change` and `misrouted`;
- per-rule FIFO;
- wait-priority after becoming eligible;
- discarding active patches after use;
- post-commit consumption and crash recovery;
- forced processing of 1–3 tail evidences at end of training;
- a trigger repeated-evidence gate and orthogonality check;
- no hard character cap on instructions; `instruction_warning_chars` (8,000 in
  v1.1) is only a soft warning/audit;
- `no_op_consumed` does not bump the revision.

When analyzing experiment results, still note:

- a rule's update count is still affected by the real trigger-activation
  frequency and candidate-production rate in the data;
- a global rule is checked on every prompt, but only enters the queue when it
  produces a valid candidate;
- the scheduler guarantees wait fairness among eligible rules, not that a rule
  which never reaches 4 updates in a normal Wave — that is handled by
  `final_flush`;
- `base_rule_hash` is currently only recorded, not yet used to re-validate stale
  patches;
- `confidence` is currently used only for auditing, not scheduling;
- after a patch leaves the active queue, its full evidence remains in the item
  `status.json`, `trajectory.json`, `selected_rule_batch.json`, and the batch
  audit.

## 16. Code index

- `vista/core/prompt_rewriter.py`: rewrite, activation filtering, global-rule
  injection, negative-terms restriction.
- `prompts/one_pass_rewrite.txt`: one-pass rewrite and `activated_rule_ids` output
  contract.
- `vista/core/agent.py`: two-round video search, learning-episode construction,
  multi-rule assessment calls.
- `scripts/run_training_corpus.py`: item status persistence, failure retry,
  distillation-only recovery.
- `vista/core/distillation.py`: per-rule assessment, `misrouted` normalization,
  batch merge, validation, and atomic commit.
- `prompts/trajectory_rule_assessments.txt`: the four-class assessment evidence
  contract.
- `prompts/batch_experience_distillation.txt`: the merge contract for 1–4
  same-rule patches.
- `vista/core/trigger_contract.py`: trigger length, global boundary, and
  conditional-trigger orthogonality checks.
- `scripts/run_batch_skill_controller.py`: Wave freezing, patch collection, FIFO
  pool, scheduling, consumption, recovery, and finalization.
- `scripts/supervise_batch_skill_controller.py`: completion conditions for
  training prompts and finalization.
- `skills/rewrite-t2v-prompt/initializations/posneg_v1_1.json`: revision-0 rules,
  trigger/update policy, and each module's initial content.
