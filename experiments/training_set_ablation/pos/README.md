# Positive-only inference ablation

`sana_ours_wo_neg` still rewrites the positive prompt with the trained Skill but
discards the learned negative prompt, keeping only the fixed negative prompt
shared by every method. It generates 495 SANA core VBench videos.

```bash
GPU_IDS=0,1,2,3,4,5,6,7 ./run.sh all
```
