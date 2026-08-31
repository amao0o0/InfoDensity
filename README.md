# InfoDensity

Official code and models for **InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning**.

Chengwei Wei, Jung-jae Kim, Longyin Zhang, Shengkai Chen, Nancy F. Chen
Institute of Advanced Intelligence and Computing (IAIC) and Centre for Frontier AI Research (CFAR), A\*STAR, Singapore

Large reasoning models spend a large fraction of their tokens on low-information deliberation.
InfoDensity is a reinforcement-learning reward that scores a reasoning trace by how *information-dense*
it is — combining an entropy-trajectory quality term with a group-relative length scaling term, applied
only to traces that reach a correct answer. Training with it produces models that keep or improve
accuracy while roughly halving the length of their reasoning traces.

## Released models

| Model | Base model | Download |
|---|---|---|
| InfoDensity-DeepSeek-R1-Distill-Qwen-7B | [`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | [🤗 amao0o0/InfoDensity-DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/amao0o0/InfoDensity-DeepSeek-R1-Distill-Qwen-7B) |
| InfoDensity-Qwen3-8B | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | [🤗 amao0o0/InfoDensity-Qwen3-8B](https://huggingface.co/amao0o0/InfoDensity-Qwen3-8B) |

## Results

Accuracy (pass@1) and mean generated length, averaged over AMC23, AIME24, MATH500 and GPQA-Diamond,
as reported in the paper (Table 2). AES is the Accuracy–Efficiency Score (α=1, β=3, γ=5).

| Model | Accuracy | Length | AES |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-7B | 58.1 | 8.5k | — |
| + InfoDensity | **72.2** | **4.5k** | **+1.20** |
| Qwen3-8B | 74.0 | 8.9k | — |
| + InfoDensity | **78.1** | **4.2k** | **+0.69** |

Per-benchmark evaluation summaries for the released Qwen3-8B checkpoint are in [`results/`](results/).

## Installation

```bash
pip install -r requirements.txt
```

## Evaluation

Evaluate a released model on the four benchmarks:

```bash
python eval/eval_vllm.py \
    --model_path amao0o0/InfoDensity-Qwen3-8B \
    --benchmark math500 \
    --temperature 0 \
    --max_tokens 16384 \
    --output_dir results/my_run
```

Repeat with `--benchmark amc23 / aime24 / gpqa`, then score the run:

```bash
python eval/compute_aes.py results/my_run --base Qwen/Qwen3-8B
```

Evaluation follows the protocol used in the paper: greedy decoding, pass@1, and the prompt
`Please reason step by step, and put your final answer within \boxed{}`.

To evaluate your own LoRA checkpoint instead, merge it into its base model first:

```bash
python eval/merge_lora.py \
    --lora_path     /path/to/checkpoint/lora_adapter \
    --tokenizer_dir /path/to/checkpoint/huggingface \
    --output_dir    /path/to/merged_model
```

`eval/run_eval.sh` drives merge → evaluate over a list of checkpoints and benchmarks.

## Repository layout

```
eval/
  eval_vllm.py           offline vLLM generation + local scoring
  grader.py              answer-equivalence checking
  math_normalize.py      answer normalization
  merge_lora.py          merge a LoRA adapter into its base model
  run_eval.sh            merge → evaluate driver
  compute_aes.py         Accuracy–Efficiency Score
  four_cell_analysis.py  preserved / rescued / still-wrong / regression breakdown
results/                 per-benchmark evaluation summaries
```

## Training

Training code is released separately and will be added to this repository.

## Acknowledgements

The evaluation pipeline builds on [ETR](https://github.com/Xuan1030/ETR) (Apache-2.0):
`eval_vllm.py` is an offline-vLLM rewrite of ETR's evaluation script, and `grader.py` is taken from it.
`math_normalize.py` derives from the Hendrycks et al. MATH release. Training used
[verl](https://github.com/volcengine/verl) (Apache-2.0). Benchmarks: AMC23, AIME24,
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) and GPQA-Diamond.

## Citation

```bibtex
@inproceedings{wei2026infodensity,
  title     = {InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning},
  author    = {Wei, Chengwei and Kim, Jung-jae and Zhang, Longyin and Chen, Shengkai and Chen, Nancy F.},
  year      = {2026}
}
```

## License

Apache-2.0. The released model weights inherit the licenses of their base models:
DeepSeek-R1-Distill-Qwen-7B (MIT) and Qwen3-8B (Apache-2.0).
