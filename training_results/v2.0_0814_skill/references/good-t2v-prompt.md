# What makes a good T2V prompt

A good text-to-video prompt is not the longest prompt and not a list of cinematic
adjectives. It is a compact, executable specification whose requested semantics can be
rendered by a finite-duration video model. Its primary objective is source fidelity;
visual polish is useful only when it does not compete with the requested action.

## Prompt rubric

Score each applicable dimension 0, 1, or 2. A deployment rewrite should not trade a
source-fidelity score for style. `2` means explicit and executable, `1` means present
but underspecified, and `0` means missing or contradictory.

| Dimension | A score of 2 requires |
|---|---|
| Intent and entity fidelity | Every requested subject, count, attribute, relation, and goal is preserved. |
| Action and temporal order | The action has a visible start, transition, and result that fit the clip. |
| Spatial grounding | Subject, reference object, and changing relation are unambiguous. |
| Identity/cardinality | Persistent entities remain the same and object counts do not drift. |
| Motion/physics | Forces, contact, direction, speed, and material response are observable and plausible. |
| Camera intent | One coherent shot, framing, focus, and movement support the action. |
| Scene complexity | The number of entities, events, and cuts is feasible for the duration. |
| Visual delivery | No unrequested text or audio; style and lighting do not obscure content. |
| Model executability | Concrete verbs/nouns replace vague goals such as “make it amazing”. |

## A useful internal representation

Before writing prose, represent a request as:

```text
subjects: [entity, count, persistent attributes]
setting: [place, time, lighting, style if requested]
initial_state: [where subjects are and what they are doing]
events: [subject, verb, object, order, visible result]
relations: [subject, reference, initial -> final]
camera: [shot, viewpoint, movement, focus]
constraints: [duration, text/audio, safety, format]
```

This representation is an audit aid, not output. It prevents a fluent rewrite from
dropping a small but load-bearing fact.

## Source-triggered modules

### Composition and focus

When there are multiple subjects or foreground/background roles, name the primary visual
center and preserve every requested component. Separate entities spatially and keep the
background uncluttered. Do not introduce crowds, decorative props, or secondary actions.

### Identity and cardinality

When a source names multiple instances, bind each action to an instance and preserve
counts. Give a persistent subject one or two stable visual identifiers only when needed
for continuity. Avoid “various”, “many”, or “a group” when the source gives a number.

### Spatial transition

When a relation changes, state subject, reference, initial relation, transition, and
final relation in that order. Keep both entities visible. Do not add detours or a cut
that makes the requested relation ambiguous.

### Temporal causality

For two or more ordered events, write a beginning, a transition, and an observable result.
Keep every verb attached to its original subject. Do not insert an unrequested outcome or
turn a causal event into a disconnected montage.

### Human motion

Use the minimum sufficient pose and motion detail. Keep limbs visible and attached,
prefer ordinary movement, and avoid rapid cuts or simultaneous gestures unless requested.
Name contact with the ground or an object when it is needed to make propulsion and
interaction clear.

### Material and physics

For force, collision, deformation, breakage, or mechanical motion, make only the source-
implied cause, direction, process, and result explicit. Preserve object identity and
material response. Do not assert physically impossible terminal states as if they were
ordinary realism.

### Camera intent

Preserve an explicitly requested shot, viewpoint, direction, and speed. If no camera is
specified, choose one restrained shot that keeps the main action legible. Never stack an
orbit, zoom, crane, rack focus, and cut into a five-second clip without a source trigger.

### Visual-only delivery

Do not add subtitles, captions, logos, dialogue, narration, music, sound effects, or
voice-over unless explicitly requested. “Cinematic”, “high quality”, and “engaging” are
not success criteria until translated into visible choices such as composition, light,
pace, or focus—and those choices must remain subordinate to intent.

## Complexity budget for short clips

For a 4–8 second clip, default to one scene, one primary action, one supporting
interaction, and zero cuts. Two or more scenes require an explicit source request and a
clear time allocation. Prefer a physically complete short action over a crowded list of
events. If the user asks for a complex story, preserve its most explicit events and make
the temporal compression visible rather than inventing more detail.

## Final audit questions

1. Can a judge point to every requested entity and action in the generated frames?
2. Does each action have a visible cause, motion, and result rather than a static pose?
3. Do counts, identities, relations, and event order remain stable?
4. Is the camera helping the action instead of becoming a second task?
5. Would removing an added adjective change what is rendered? If not, remove it.
6. Did the rewrite add any unrequested text or audio instruction? If yes, remove it.
