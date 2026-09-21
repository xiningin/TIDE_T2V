# EvalCrafter 100-prompt Skill training selection

## 1. Files

- Source corpus: `training_prompts/evalcrafter_prompt700.txt`
- Selection result: `training_prompts/evalcrafter_prompt100_skill_train.txt`
- Traceable metadata: `training_prompts/evalcrafter_prompt100_skill_train.selection.json`
- Selection size: 100 prompts
- Uniqueness: all 100 remain unique after case normalization

The original 700-prompt file is unchanged. Text in the selection keeps its
original wording, with only leading/trailing whitespace and duplicate spaces
cleaned up; every prompt is traceable to its original line number via the
selection JSON.

## 2. Selection goal

These 100 prompts are not meant to reproduce EvalCrafter's full distribution;
they build a small, high-information-density subset suited to training the TIDE
positive/negative prompt-rewrite Skill. Selection considered:

1. Video content should be visible, generatable, and evaluable by a
   video-understanding model, yielding clear generation feedback.
2. Prompts should give a short clip a clear subject, action, or scene while
   leaving room for rewriting.
3. Cover the most common training signals in the current Skill: action physics,
   identity/count, scene completion, camera presentation, and environmental
   dynamics.
4. Include both easy and hard cases, so the Skill learns not only from complex
   prompts but also from short, under-specified, or tightly constrained inputs.
5. Reduce interference from duplicates, pure text rendering, celebrity-face
   descriptions, and style-word stacking.

## 3. Categories and rationale

| Curated category | Count | Rationale | Main training value |
|---|---:|---|---|
| Action & physics | 35 | Displacement, rotation, contact, force, stopping, landing, continuous manipulation, material response | Action timelines, trajectories, contact, speed changes, causal order, visible end states |
| Count & identity | 25 | Two+ entities, exact counts, multi-agent cooperation, look-alike persistent subjects | Count preservation, subject separation, identity consistency; fewer add/drop/fuse/duplicate/morph failures |
| Camera & subject presentation | 15 | Close-up, wide, handheld, pan, zoom, rear view, explicit subject appearance | Turning camera requirements and subject info into executable, coherent shots |
| Scene completion & weak action | 15 | Static objects, quiet scenes, subtle natural motion, short descriptions | Whether the Skill completes needed environment/framing/micro-motion without unsupported expansion |
| Environmental dynamics | 10 | Rain, fog, wind, flowing water, dunes, smoke, thunderstorms — non-subject motion | Environmental change, particle motion, lighting response, separating background dynamics from subject action |

These categories curate the data distribution; they are not the rules that must
fire at training time. Actual rules depend on the Skill trigger, the rewrite
trajectory, and the video feedback, and one prompt may activate several rules.

## 4. Concrete inclusion criteria

### 4.1 Observable and evaluable

Prefer requests whose success can be judged directly from the video, e.g.:

- A horse galloping and kicking up dust — check trajectory, gait, environment
  response;
- A chef continuously peeling potatoes — check hand/tool/object contact and
  continuity;
- A hockey player stopping abruptly and spraying ice — check deceleration,
  stopping, material response;
- A specified number of animals or people — check count preservation and whether
  fusion, duplication, or disappearance occurs.

These prompts make it easier for the Qwen critique to produce feedback tied
directly to rewriting behavior, and for DeepSeek to distill reusable patches
from the full trajectory.

### 4.2 Learnable failure modes

The selection deliberately keeps cases likely to expose model problems:

- Large motion, fast turns, landings, stops;
- Multi-subject cooperation, similar subjects, exact counts;
- Small-object manipulation, tool contact, continuous hand motion;
- Continuous environmental change (wind, rain, fog, smoke);
- Scenes with simultaneous camera and subject motion.

These produce relatively clear negative phenomena — trajectory drift, broken
contact, identity switching, count drift, subject fusion, framing drift, and
temporal discontinuity — good for training conditional negative prompts.

### 4.3 Difficulty/information balance

The data includes both fully described prompts and a few short-but-valid
prompts, e.g. `Thunderstorm at night`, `Boats on the calm, blue ocean`, and
`a clock with no hands`.

These short cases test whether the Skill can:

- Add the minimal scene information needed for generation;
- Keep the original subject and core semantics;
- Avoid adding unrelated events just because the input is short;
- Handle explicit negation ("without X") correctly rather than pushing the target
  semantics into the negative prompt.

## 5. Exclusions and why

The following were not used as the main training set:

- **Exact or near-duplicate prompts**: they waste expensive video-generation
  budget and inflate the update frequency of local rules.
- **Pure text/sign/word-rendering tasks**: they mainly test OCR/text generation,
  misaligned with this Skill's subject/action/temporal optimization goal.
- **Long runs of celebrity facial-expression descriptions**: highly homogeneous;
  they bias training toward specific people and face close-ups.
- **Heavy artist/style-word stacking**: failures may come from style reproduction
  rather than prompt structure, hard to attribute to reusable rules.
- **Isolated nouns with no visible relations**: usually lack a stable action or
  temporal learning signal.
- **Pure audio goals**: the current evaluation chain is visual-frame based and
  unsuited to verifying singing, sound, or music quality.
- **Semantically contradictory or badly broken grammar**: they mix input errors
  with generation errors in the critique, lowering patch causal reliability.

## 6. Relationship to the current Skill structure

This subset mainly provides training opportunities for:

- `vista.intent_preservation`: every prompt must preserve subject, count, action,
  and explicit constraints.
- `vista.scene_components`: short or static prompts test necessary
  environment/framing/lighting/micro-motion completion.
- `vista.motion_physics`: action, interaction, stop, landing, and environment
  response cases provide the main physical feedback.
- `vista.identity_count_guard`: exact-count and multi-subject cases provide
  identity/cardinality preservation signal.
- `vista.visual_only_delivery`: most prompts have no explicit audio/text
  requirement, testing against wrongly added captions, narration, or visible
  text.

The original EvalCrafter corpus has almost no explicit duration constraints and
lacks complete "leave frame then re-enter" descriptions, so this subset covers
these rules only lightly:

- `vista.duration_scene_budget`
- `vista.frame_exit_reentry`

To train all rules more evenly later, add targeted prompts with explicit
durations, single-shot/multi-scene budgets, and exit/occlusion/re-entry
processes, rather than forcing mismatched samples out of the existing 700.

## 7. Ordering

The output file is not split into five contiguous category blocks; it is
interleaved, in order to:

- Avoid several consecutive waves seeing only one task type;
- Reduce one prompt category dominating the patch pool early in training;
- Bring different failure modes into the experience pool sooner.

Interleaving does not guarantee perfectly even rule updates. Which rule TIDE
ultimately selects still depends on how trajectory patches are classified and on
the accumulation of four same-rule patches.

## 8. Validation

- Non-empty prompts: 100
- Unique prompts after case normalization: 100
- Unique original line numbers: 100
- Sum across the five categories: 100
- All output text matches the whitespace-normalized text of its original line
- Output SHA256: `0bcd0be07e5bc412b96a102fe30137ddbeb7037d613b6777d5fa526615b9b910`

See `evalcrafter_prompt100_skill_train.selection.json` for the detailed category
list, output order, and original line numbers.
