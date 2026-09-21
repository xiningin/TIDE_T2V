# Random-trigger ablation training time

Data source: 100 `run/stats.json` files, 13 `batch_manifest.json` files, the
controller log, and 17 `batch_distillation/attempt_01.json` files under
`outputs/ablation_random_trigger/`.

## Overall timeline

- GPUs: 4, 5, 6, 7 (one SANA worker per shard)
- Data: 100 training prompts, 13 waves (first 12 waves of 8 prompts each, last
  wave of 4)
- First wave start: 2026-08-26 05:52:54 UTC
- Last training wave finished: 2026-08-26 19:53:49 UTC
- Final rule-pool finalization finished: 2026-08-26 19:56:00 UTC
- End-to-end wall-clock time: 50,585.6 s = **14 h 03 m 06 s**
- Sum of the 13 training-wave wall-clock times: 50,454.6 s = **14 h 00 m 55 s**
- Final rule-pool finalization: ~130.8 s (~2 m 11 s)

## VISTA search trajectory time

Every prompt runs a two-round VISTA-style search: planning, initial raw/skill
videos, visual critique, DTPA optimization, candidate video generation, and
tournament selection. Each trajectory produces 10 videos (100 prompts = 1,000
T2V calls). The table below sums the per-stage time across all 100 episodes;
this is the sum of serial time across GPU workers, not wall-clock time.

| Stage | Cumulative worker seconds | Cumulative hours | Share of episode stage time |
|---|---:|---:|---:|
| Planning | 872.5 | 0.24 h | 0.48% |
| T2V generation (1,000 videos) | 129,967.4 | 36.10 h | 71.14% |
| Visual critique | 19,181.8 | 5.33 h | 10.50% |
| DTPA optimization | 10,365.4 | 2.88 h | 5.67% |
| Tournament selection | 18,393.0 | 5.11 h | 10.07% |
| **VISTA search subtotal** | **178,780.1** | **49.66 h** | **97.86%** |
| Trajectory distillation (compress each trajectory into a candidate patch) | 3,916.1 | 1.09 h | 2.14% |
| **100-episode total** | **182,696.2** | **50.75 h** | **100%** |

With four GPUs in parallel, the cumulative worker time of the VISTA search stage
corresponds to roughly 12.42 hours of ideal parallel compute; the actual 13-wave
wall-clock time is about 14.01 hours, with the gap coming from model loading,
inter-wave scheduling, API waits, and load imbalance.

## Actual Skill-update time

This experiment does not train SANA or any other model weights. The SANA weights
stay frozen; the Skill is updated through DeepSeek-V4-Flash rule distillation:

- 12 in-wave rule batch distillations: ~**300.7 s (5 m 01 s)** total
- 5 finalization rule updates after all prompts complete: ~**130.2 s (2 m 10 s)**
  total
- 17 distillation API calls combined: ~**430.9 s (7 m 11 s)**
- Final random-trigger Skill revision: 0 → 16 (one of which was a no-op consumed)

So if "actual training" means the time spent modifying Skill rules, the answer is
about **7 minutes**; if it means producing the VISTA search trajectories that
serve as training evidence, it is about **14 hours** of wall-clock time
(cumulative worker stage time ~49.66 hours, of which ~36.10 hours is T2V
generation). If it means neural-network parameter training, this experiment is
**0 hours** — there are no gradient updates.
