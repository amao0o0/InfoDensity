# Evaluation summaries

Raw per-benchmark summaries written by `eval/eval_vllm.py` (greedy decoding, pass@1,
16384-token generation cap).

| Directory | What it is |
|:--|:--|
| `infodensity_qwen3_8b/` | [InfoDensity-Qwen3-8B](https://huggingface.co/amao0o0/InfoDensity-Qwen3-8B) |
| `infodensity_dsr1_7b/` | [InfoDensity-DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/amao0o0/InfoDensity-DeepSeek-R1-Distill-Qwen-7B) |
| `qwen3_8b_base/` | Our own run of the unmodified `Qwen/Qwen3-8B`, for reference |

Each file records accuracy, number of correct answers and mean generated length for one benchmark.
Per-sample generations are not included here because of their size; re-running `eval/eval_vllm.py`
regenerates them.

## A note on reproducibility

Greedy decoding is not bit-reproducible under vLLM: batching and kernel selection perturb
floating-point rounding, which occasionally flips a token and sends a trace down a different path.
Expect run-to-run variation of roughly ±1 accuracy point on MATH500 and ±1–2 points on
GPQA-Diamond, whose 198 questions make each one worth 0.5 points. Mean accuracy over the four
benchmarks is considerably more stable than any single benchmark.

The baseline rows in the paper's Table 2 for the 7B and 8B blocks are quoted from prior work rather
than re-run here, so `qwen3_8b_base/` may differ from Table 2 by a small margin.
