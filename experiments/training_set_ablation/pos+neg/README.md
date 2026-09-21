# Positive + negative (canonical ours)

This arm is exactly the main `sana_ours` experiment; do not generate a second
copy of the videos. Launch it from `../../cross_model_results/sana-2b/`:

```bash
cd ../../cross_model_results/sana-2b
GPU_IDS=0,1,2,3,4,5,6,7 ./run.sh sana_ours all
```

The canonical outputs live in `cross_model_results/sana-2b/sana_ours/`.
