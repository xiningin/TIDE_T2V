---
name: rewrite-t2v-prompt
description: One-pass, test-time text-to-video positive and conditional-negative prompt rewriting with a source-preserving semantic audit and a reusable self-improvement contract. Use when converting a user idea into a detailed generation-ready video prompt pair or distilling accepted visual-feedback episodes into a bounded prompt-skill patch.
---

# T2V Prompt Rewrite Skill

Rewrite a user request into one generation-ready positive prompt and one compact conditional negative prompt. The deployed
path is deliberately one-pass: `user request -> rewrite -> T2V generator`. Do not
generate candidate videos, call a video judge, run a tournament, or expose chain of
thought during deployment.

## Mode selection

- **Rewrite mode (default):** If the input is a user request and optionally an existing
  T2V prompt, return one structured positive/negative pair.
- **Patch mode (training only):** If the input is a structured accepted/rejected
  episode containing the source prompt, rewrite, visual feedback, outcome scores, and
  held-out evidence, return one bounded JSON skill patch. Never mutate the skill from a
  single episode; promotion is an external, held-out-gated operation.

## One-pass rewrite procedure

Perform the following privately; output none of the intermediate reasoning.

1. **Build a fidelity ledger.** Extract every explicit entity, count, attribute,
   relation, action, event order, camera instruction, style, quoted text, duration,
   and delivery constraint. Mark each as immutable. Preserve the user's intent even
   when the source is short or ungrammatical.
2. **Compile a feasible action graph.** For a short clip, use at most one main subject
   and one coherent action arc unless the source requires more. Express ordered events
   as beginning -> transition -> observable result. Do not add an event merely to make
   the prompt sound cinematic.
3. **Activate only source-triggered modules.** Apply the modules in `rules.json`.
   Treat
   `trigger_axis` as the module's sole routing responsibility. Multiple modules may
   activate when separate source facts independently satisfy separate axes; do not
   activate one module merely because another module fired or because it is
   aesthetically attractive.
4. **Compile the positive prompt.** Prefer detailed, observable specifications over vague brevity: bind each action to its subject; specify source-triggered screen position, initial pose, motion direction/path, contact, force, material response, and end state. Describe exit and re-entry edges with identity continuity when requested or implied by the action. Detail must remain executable within the duration and may not introduce a competing event.
5. **Compile the conditional negative prompt.** Select at most eight short artifact phrases from activated rules. Use failure concepts rather than desired entity names. A fixed global negative prompt is merged downstream, so do not repeat generic terms.
6. **Run the silent verifier.** Check source coverage, entity counts, action order,
   spatial relations, duration/action budget, camera consistency, physics, and unwanted
   text/audio, and negative-to-positive conflicts. If any edit fails, revert it.

## Non-negotiable preservation rules

- Do not delete, weaken, reassign, reorder, or contradict an explicit source fact.
- Do not invent people, objects, actions, dialogue, captions, music, sound effects,
  voice-over, logos, or scene cuts unless requested or strictly necessary to express
  the source.
- Do not turn a requested action into a static pose, unspecified montage, glide without
  propulsion, or a sequence that cannot plausibly fit the duration.
- Do not add a second camera move when the source specifies a camera; preserve a static
  camera when the source is static.
- If a requested effect is physically impossible, preserve the intent and make the
  smallest visual interpretation that is consistent with the stated style (for
  example, animation remains animation). Do not silently change the subject or goal.

## Batch-learning contract

Batch learning is training-only. Compress each complete trajectory into at most one
patch for each activated rule. Keep unused patches in a persistent cross-Wave pool.
In each Wave select at most one rule; combine same-rule patches and update only when
four are available. Consume those patches only after a successful atomic replacement,
leaving every other rule unchanged. Preserve `trigger` by default; a trigger change
requires repeated routing evidence, at least two compatible supporting patches, and
the conditional-trigger orthogonality audit. `trigger_axis` and `capability_scope`
remain immutable.

## Output contract

Rewrite mode: one JSON object with `positive_prompt` and `negative_prompt` and nothing else.

Batch-learning mode: one JSON object containing a replacement for the selected rule only.
