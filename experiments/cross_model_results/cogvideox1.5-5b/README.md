# CogVideoX1.5-5B generation

Run `cogvideox15_raw` and `cogvideox15_ours`, 1,293 VBench videos each.

```bash
GPU_IDS=0,1,2,3,4,5,6,7 ./run.sh cogvideox15_raw all
GPU_IDS=0,1,2,3,4,5,6,7 ./run.sh cogvideox15_ours all
```

Outputs land in `cogvideox15_raw/` and `cogvideox15_ours/` respectively;
evaluation is not triggered automatically.
