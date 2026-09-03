# Evaluation summaries

Raw per-benchmark summaries written by `eval/eval_vllm.py` (greedy decoding, pass@1,
16384-token generation cap).

| Directory | What it is |
|:--|:--|
| `infodensity_qwen3_8b/` | [InfoDensity-Qwen3-8B](https://huggingface.co/amao0o0/InfoDensity-Qwen3-8B) |
| `infodensity_dsr1_7b/` | [InfoDensity-DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/amao0o0/InfoDensity-DeepSeek-R1-Distill-Qwen-7B) |
| `infodensity_dsr_llama_8b/` | [InfoDensity-DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/amao0o0/InfoDensity-DeepSeek-R1-Distill-Llama-8B) |
| `qwen3_8b_base/` | Our own run of the unmodified `Qwen/Qwen3-8B`, for reference |

Each file records accuracy, number of correct answers and mean generated length for one benchmark.
Per-sample generations are not included here because of their size; re-running `eval/eval_vllm.py`
regenerates them.

## A note on reproducibility

Greedy decoding is not bit-reproducible under vLLM: batching and kernel selection perturb
floating-point rounding, which occasionally flips a token and sends a trace down a different path.

How much that matters depends on the benchmark and the model. `infodensity_dsr_llama_8b/` was
produced by downloading the model from the Hub and re-running the pipeline months after the numbers
in the paper were measured, on different hardware, and AMC23, AIME24 and MATH500 came back
**byte-identical** — every one of the 40, 30 and 500 generations. So these numbers are not fragile
in general.

They can be fragile in particular, though, and the risk concentrates in two places. AMC23 and
AIME24 have only 40 and 30 problems, so a single flipped problem moves them by 2.5 and 3.3 points.
And a trace that forks mid-generation can fall into a repetition loop and run to the 16384-token
cap, where it produces no boxed answer and scores zero — so one perturbation can cost a whole
problem rather than a rounding error. Smaller models with longer traces are the most exposed.

Mean accuracy over the four benchmarks is considerably more stable than any single benchmark; treat
individual cells as point estimates rather than exact quantities.

The baseline rows in the paper's Table 2 for the 7B and 8B blocks are quoted from prior work rather
than re-run here, so `qwen3_8b_base/` may differ from Table 2 by a small margin.

GPQA-Diamond is scored with the multiple-choice fallback described in the top-level README, on both
the model and its base.
