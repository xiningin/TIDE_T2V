# VPO generation

`cogvideox15_vpo` first rewrites the 431 VBench prompts in batch with a local
VPO-5B checkpoint, then generates 1,293 paired videos with CogVideoX1.5-5B.

Point `VPO_MODEL_DIR` at your local VPO-5B weights (see
`../../benchmark_eval/config/generation_registry.json`) before launching.

```bash
REWRITE_GPU=0 GPU_IDS=0,1,2,3,4,5,6,7 ./run.sh all
```

If prompt preparation and video generation cannot share the GPUs, run the two
stages separately:

```bash
REWRITE_GPU=0 ./run.sh prepare
GPU_IDS=0,1,2,3,4,5,6,7 ./run.sh generate
./run.sh audit
```
