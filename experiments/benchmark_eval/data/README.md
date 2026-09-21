# Data provenance

The files under this directory are pinned local copies for the curated
evaluation and should not be hand-edited.

| Local directory | Upstream source |
|---|---|
| `vbench/prompts/` | `VBench/prompts/prompts_per_dimension/` |
| `vbench/metadata/` | `VBench/prompts/metadata/` and `VBench/vbench/VBench_full_info.json` |
| `t2v_compbench/prompts/` | `T2V-CompBench/prompts/` |
| `t2v_compbench/metadata/` | `T2V-CompBench/meta_data/` |

For the CompBench numeracy split, five prompts differ from the metadata only by
a trailing period; the task list is generated from the prompt TXT files while
the metadata keeps the official originals, and the two share the same index
order.

The license, citation, and full implementation of each benchmark follow the
official upstream repositories (VBench-2.0 and T2V-CompBench).
