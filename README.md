<div align="center">

# InfoDensity

**Rewarding Information-Dense Traces for Efficient Reasoning**

[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Main-b31b1b.svg)](https://2026.emnlp.org/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-InfoDensity-ffbd45.svg)](https://huggingface.co/amao0o0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Chengwei Wei · Jung-jae Kim · Longyin Zhang · Shengkai Chen · Nancy F. Chen

*Institute of Advanced Intelligence and Computing (IAIC) · Centre for Frontier AI Research (CFAR)*
*A\*STAR, Singapore*

</div>

---

Large reasoning models spend much of their token budget on low-information deliberation —
re-deriving results they already have, or looping until they exhaust their generation cap.

**InfoDensity** is a reinforcement-learning reward that scores a reasoning trace by how
*information-dense* it is. It combines an entropy-trajectory quality term with a group-relative
length scaling term, and applies the product only to traces that reach a correct answer:

$$R_{\text{InfoDensity}}(\tau) = R_{\text{quality}}(\tau) \cdot R_{L}(\tau)$$

Training with it keeps or improves accuracy while roughly halving the length of the reasoning trace.

## 🤗 Models

| Model | Base | Accuracy | Length | AES |
|:--|:--|:--:|:--:|:--:|
| [**InfoDensity-Qwen3-8B**](https://huggingface.co/amao0o0/InfoDensity-Qwen3-8B) | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 78.1 <sub>(+4.1)</sub> | 4.2k <sub>(−53%)</sub> | **+0.69** |
| [**InfoDensity-DeepSeek-R1-Distill-Qwen-7B**](https://huggingface.co/amao0o0/InfoDensity-DeepSeek-R1-Distill-Qwen-7B) | [DSR-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | 72.2 <sub>(+14.1)</sub> | 4.5k <sub>(−47%)</sub> | **+1.20** |
| [**InfoDensity-DeepSeek-R1-Distill-Llama-8B**](https://huggingface.co/amao0o0/InfoDensity-DeepSeek-R1-Distill-Llama-8B) | [DSR-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) | 58.4 <sub>(+10.8)</sub> | 6.5k <sub>(−31%)</sub> | **+0.99** |

Accuracy and length are means over AMC23, AIME24, MATH500 and GPQA-Diamond; deltas are relative to
the untrained base model. AES is the Accuracy–Efficiency Score (α=1, β=3, γ=5).

Raw per-benchmark summaries for every released checkpoint are in [`results/`](results/).

<details>
<summary><b>Per-benchmark results (paper, Table 2)</b></summary>

| Model | AMC23 | AIME24 | MATH500 | GPQA-D | Overall |
|:--|:--:|:--:|:--:|:--:|:--:|
| DeepSeek-R1-Distill-Qwen-7B | 80.0 / 6.6k | 43.3 / 11.8k | 85.0 / 4.2k | 24.2 / 11.3k | 58.1 / 8.5k |
| **+ InfoDensity** | 95.0 / 3.6k | 60.0 / 6.5k | 91.0 / 2.1k | 42.9 / 5.6k | **72.2 / 4.5k** |
| Qwen3-8B | 90.0 / 8.0k | 63.3 / 12.2k | 90.6 / 5.4k | 52.0 / 9.9k | 74.0 / 8.9k |
| **+ InfoDensity** | 90.0 / 3.7k | 73.3 / 6.9k | 93.0 / 2.1k | 56.1 / 4.2k | **78.1 / 4.2k** |

And DeepSeek-R1-Distill-Llama-8B (paper, Appendix E / Table 4):

| Model | AMC23 | AIME24 | MATH500 | GPQA-D | Overall |
|:--|:--:|:--:|:--:|:--:|:--:|
| DeepSeek-R1-Distill-Llama-8B | 65.0 / 8.5k | 23.3 / 13.3k | 78.8 / 5.2k | 23.2 / 10.7k | 47.6 / 9.4k |
| **+ InfoDensity** | 80.0 / 5.4k | 40.0 / 10.0k | 82.8 / 3.2k | 30.8 / 7.6k | **58.4 / 6.5k** |

Cells are accuracy (%) / mean response tokens.

The GPQA column credits a response that states its choice as `ANSWER: C` without `\boxed{}`;
the paper's Appendix E table counted only boxed answers, which on these same outputs gives 23.2
for the model against 17.7 for the base — 56.5 / 6.6k overall, and an essentially unchanged AES.
Both sides of every comparison here use the same rule. The other columns match the paper exactly.

</details>

## 🚀 Quick start

```bash
pip install -r requirements.txt
```

Evaluate a released model on one benchmark:

```bash
python eval/eval_vllm.py \
    --model_path amao0o0/InfoDensity-Qwen3-8B \
    --benchmark math500 \
    --temperature 0 \
    --max_tokens 16384 \
    --output_dir results/my_run
```

`--benchmark` accepts `amc23`, `aime24`, `math500` and `gpqa`. Once all four have run, score them:

```bash
python eval/compute_aes.py results/my_run --base Qwen/Qwen3-8B
```

Evaluation follows the protocol used in the paper: greedy decoding, pass@1, a 16384-token generation
cap, and the prompt `Please reason step by step, and put your final answer within \boxed{}`.

<details>
<summary><b>Evaluating your own LoRA checkpoint</b></summary>

Merge the adapter into its base model first — the base model id is read from the adapter's own
`adapter_config.json`:

```bash
python eval/merge_lora.py \
    --lora_path     /path/to/checkpoint/lora_adapter \
    --tokenizer_dir /path/to/checkpoint/huggingface \
    --output_dir    /path/to/merged_model
```

`eval/run_eval.sh` drives merge → evaluate over a list of checkpoints and benchmarks in one pass.

</details>

## 📁 Repository layout

```
eval/
├── eval_vllm.py           offline vLLM generation + local scoring
├── grader.py              answer-equivalence checking
├── math_normalize.py      answer normalization
├── merge_lora.py          merge a LoRA adapter into its base model
├── run_eval.sh            merge → evaluate driver
├── compute_aes.py         Accuracy–Efficiency Score
└── four_cell_analysis.py  preserved / rescued / still-wrong / regression breakdown
training/
├── infodensity_reward.py  the reward: correctness gate x R_quality x R_L
├── entropy_trajectory.py  suffix-max envelope and the R_quality term
├── prepare_data.py        DeepMath-103K subset -> verl parquet
└── verl_integration.md    forwarding rollout entropy to the reward function
results/                   per-benchmark evaluation summaries
```

## 🏋️ Training

[`training/`](training/) holds the reward and the data it trains on. The RL loop
itself is [verl](https://github.com/volcengine/verl); the reward plugs into its
`batch` reward manager.

$$R_{\text{InfoDensity}}(\tau) = R_{\text{quality}}(\tau)\cdot R_{L}(\tau)\quad\text{if }\tau\text{ is correct},\qquad 0\ \text{otherwise.}$$

The correctness gate comes first, so neither factor can buy brevity with accuracy.

- **$R_{\text{quality}}$** scores information density from the trace's own rollout
  entropy — no judge model and no second forward pass. Per-token entropy is chunked
  and averaged into a trajectory, replaced by its **suffix-max envelope**
  $E_t = \max(H_t,\dots,H_T)$ so that trailing high-entropy filler propagates
  backward instead of hiding behind a confident prefix, then normalised by
  $\log K$ — a constant shared by every trace, so no trace can improve its score by
  inflating its own reference point.
- **$R_L$** scales a correct trace by $\exp(-\lambda z)$, where $z$ is its length
  z-score among the *correct* traces sampled for the same prompt. Comparing
  within a prompt avoids holding easy and hard problems to one length budget, and
  the group mean re-centres as the policy shortens.

Training with it needs one change to verl so that rollout entropy reaches the
reward function — without it `R_quality` is silently skipped and only the length
term trains. [`training/README.md`](training/README.md) walks through data prep,
that patch, and the hydra overrides; hyperparameters are arguments there, not
constants, so set them for your own setup.

## 🙏 Acknowledgements

The evaluation pipeline builds on [ETR](https://github.com/Xuan1030/ETR) (Apache-2.0): `eval_vllm.py`
is an offline-vLLM rewrite of its evaluation script, and `grader.py` is taken from it.
`math_normalize.py` derives from the Hendrycks et al. MATH release. Training used
[verl](https://github.com/volcengine/verl) (Apache-2.0).

## 📄 Citation

```bibtex
@inproceedings{wei2026infodensity,
  title     = {InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning},
  author    = {Wei, Chengwei and Kim, Jung-jae and Zhang, Longyin and Chen, Shengkai and Chen, Nancy F.},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## ⚖️ License

Apache-2.0 (see [LICENSE](LICENSE)). The released weights inherit their base models' licenses:
the DeepSeek-R1-Distill checkpoints (Qwen-7B, Llama-8B) are MIT, and Qwen3-8B is
Apache-2.0.
