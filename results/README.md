# Evaluation summaries

Raw per-benchmark summaries written by `eval/eval_vllm.py` (greedy decoding, pass@1,
16384-token generation cap).

| Directory | What it is |
|:--|:--|
| `infodensity_qwen3_8b/` | [InfoDensity-Qwen3-8B](https://huggingface.co/amao0o0/InfoDensity-Qwen3-8B) |
| `infodensity_dsr1_7b/` | [InfoDensity-DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/amao0o0/InfoDensity-DeepSeek-R1-Distill-Qwen-7B) |
| `infodensity_dsr_llama_8b/` | [InfoDensity-DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/amao0o0/InfoDensity-DeepSeek-R1-Distill-Llama-8B) |
| `qwen3_8b_base/` | The unmodified `Qwen/Qwen3-8B`, for reference |

Each file records accuracy, number of correct answers and mean generated length for one benchmark.
Per-sample generations are not included here because of their size; re-running `eval/eval_vllm.py`
regenerates them.
