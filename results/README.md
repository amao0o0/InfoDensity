# Evaluation summaries

Raw per-benchmark summaries written by `eval/eval_vllm.py` (greedy decoding, pass@1).

| Directory | What it is |
|---|---|
| `infodensity_qwen3_8b/` | The released [InfoDensity-Qwen3-8B](https://huggingface.co/amao0o0/InfoDensity-Qwen3-8B) checkpoint |
| `qwen3_8b_base/` | Our own run of the unmodified `Qwen/Qwen3-8B`, for reference |

Each file records accuracy, number of correct answers and mean generated length for one benchmark.
Per-sample generations are not included here because of their size; re-running
`eval/eval_vllm.py` regenerates them.

Note that the baseline rows in the paper's Table 2 for the 7B and 8B blocks are quoted from prior
work rather than from these runs, so `qwen3_8b_base/` may differ from Table 2 by a small margin.
