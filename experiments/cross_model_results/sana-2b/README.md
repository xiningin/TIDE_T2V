# SANA-Video 2B generation

Two directly runnable experiments live here:

- `sana_raw`: 1,293 VBench videos + 400 CompBench videos;
- `sana_ours`: same tasks and seeds, using the positive/negative prompts from the
  trained Skill (revision 12 in the recorded run).

Launch from this directory:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 ./run.sh sana_raw all
GPU_IDS=0,1,2,3,4,5,6,7 ./run.sh sana_ours all
```

Each arm is self-contained: `experiment.json`, `jobs.jsonl`, `videos/`,
`receipts/`, `logs/`, and a final `generation_audit.json`. The script only
generates and audits videos; it does not run VBench or CompBench.
