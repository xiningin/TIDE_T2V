# Conditional negative-prompt vocabulary

Use short artifact concepts, never desired entity names or full logical sentences. Select
only source-triggered terms and emit at most eight after case-insensitive deduplication.

- temporal: `temporal inconsistency`, `flicker`, `abrupt appearance`, `sudden disappearance`, `identity switching`
- count/identity: `duplicated subjects`, `missing subjects`, `subject fusion`, `merged bodies`, `count drift`
- anatomy: `extra limbs`, `detached limbs`, `fused limbs`, `broken anatomy`, `joint deformation`
- objects/contact: `duplicated objects`, `missing objects`, `object morphing`, `floating objects`, `broken contact`, `interpenetration`
- motion/physics: `sliding motion`, `reversed motion`, `trajectory drift`, `impossible deformation`, `discontinuous motion`
- camera: `camera shake`, `unintended zoom`, `abrupt cut`, `framing drift`

Do not include a requested noun such as `dog`, `sheep`, `cup`, or `car` in the negative
prompt. Do not negate a requested interaction; guard artifacts around the interaction.
