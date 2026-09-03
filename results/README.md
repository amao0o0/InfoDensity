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
Usually this is harmless — re-running the pipeline on `infodensity_dsr_llama_8b/` reproduced AMC23,
AIME24 and MATH500 byte for byte, all 40, 30 and 500 generations.

Where it does bite: AMC23 and AIME24 have only 40 and 30 problems, so one flipped problem moves
them by 2.5 and 3.3 points, and a trace that forks into a repetition loop runs to the 16384-token
cap and scores zero rather than merely rounding differently. Smaller models with longer traces are
the most exposed. Treat individual cells as point estimates; the mean over the four benchmarks is
far steadier than any one of them.

On GPQA-Diamond a response stating its choice as `ANSWER: C` without `\boxed{}` is credited, for
model and base alike.
