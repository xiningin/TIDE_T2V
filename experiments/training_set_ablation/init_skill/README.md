# Initial-Skill ablation

`sana_init_skill` reads
`skills/rewrite-t2v-prompt/initializations/posneg_v1_1.json` directly (rather
than the trained `working_skill/rules.json`) and generates 495 SANA core VBench
videos. The `initializations/` directory therefore has a concrete use in this
ablation.

```bash
GPU_IDS=0,1,2,3,4,5,6,7 ./run.sh all
```
