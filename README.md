<div align="center">

# InfoDensity

### Rewarding Information-Dense Traces for Efficient Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2603.17310-b31b1b.svg)](https://arxiv.org/abs/2603.17310)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Main-5c2d91.svg)](https://2026.emnlp.org/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-InfoDensity-ffbd45.svg)](https://huggingface.co/amao0o0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Chengwei Wei · Jung-jae Kim · Longyin Zhang · Shengkai Chen · Nancy F. Chen

*Centre for Frontier AI Research (CFAR)*<br>
*Institute of Advanced Intelligence and Computing (IAIC)*<br>
*A\*STAR, Singapore*

</div>

---

Large reasoning models generate verbose, redundant reasoning traces at real computational cost.
Reinforcement learning that optimizes only the final response length leaves the quality of the
intermediate steps unsupervised, which invites reward hacking. Verbosity is not merely a length
problem — it is a symptom of poor intermediate reasoning.

Tracking per-token predictive entropy across reasoning trajectories, we find that high-quality
traces share two properties: **low uncertainty convergence** and **fast uncertainty descent**. Such
traces are *informationally dense* — their steps reach a low uncertainty level relative to the total
reasoning length. **InfoDensity** rewards both through a single suffix-max envelope of the entropy
trajectory, weighted by a length scaling term that favours achieving equivalent quality more
concisely:

$$R_{\text{InfoDensity}}(\tau) = R_{\text{quality}}(\tau) \cdot R_{L}(\tau)$$

## 🤗 Models

<table>
<thead>
<tr><th align="left"></th><th>AMC23</th><th>AIME24</th><th>MATH500</th><th>GPQA-D</th><th>Overall</th><th>AES</th></tr>
</thead>
<tbody>
<tr><td colspan="7"><b>DeepSeek-R1-Distill-Qwen-7B</b></td></tr>
<tr><td align="left">Base</td><td align="center">80.0&nbsp;/&nbsp;6.6k</td><td align="center">43.3&nbsp;/&nbsp;11.8k</td><td align="center">85.0&nbsp;/&nbsp;4.2k</td><td align="center">24.2&nbsp;/&nbsp;11.3k</td><td align="center">58.1&nbsp;/&nbsp;8.5k</td><td align="center">—</td></tr>
<tr><td align="left"><a href="https://huggingface.co/amao0o0/InfoDensity-DeepSeek-R1-Distill-Qwen-7B"><b>+ InfoDensity</b></a></td><td align="center"><b>95.0&nbsp;/&nbsp;3.6k</b></td><td align="center"><b>60.0&nbsp;/&nbsp;6.5k</b></td><td align="center"><b>91.0&nbsp;/&nbsp;2.1k</b></td><td align="center"><b>42.9&nbsp;/&nbsp;5.6k</b></td><td align="center"><b>72.2&nbsp;/&nbsp;4.5k</b></td><td align="center"><b>+1.20</b></td></tr>
<tr><td colspan="7"><b>Qwen3-8B</b></td></tr>
<tr><td align="left">Base</td><td align="center">90.0&nbsp;/&nbsp;8.0k</td><td align="center">63.3&nbsp;/&nbsp;12.2k</td><td align="center">90.6&nbsp;/&nbsp;5.4k</td><td align="center">52.0&nbsp;/&nbsp;9.9k</td><td align="center">74.0&nbsp;/&nbsp;8.9k</td><td align="center">—</td></tr>
<tr><td align="left"><a href="https://huggingface.co/amao0o0/InfoDensity-Qwen3-8B"><b>+ InfoDensity</b></a></td><td align="center"><b>90.0&nbsp;/&nbsp;3.7k</b></td><td align="center"><b>73.3&nbsp;/&nbsp;6.9k</b></td><td align="center"><b>93.0&nbsp;/&nbsp;2.1k</b></td><td align="center"><b>56.1&nbsp;/&nbsp;4.2k</b></td><td align="center"><b>78.1&nbsp;/&nbsp;4.2k</b></td><td align="center"><b>+0.69</b></td></tr>
<tr><td colspan="7"><b>DeepSeek-R1-Distill-Llama-8B</b></td></tr>
<tr><td align="left">Base</td><td align="center">65.0&nbsp;/&nbsp;8.5k</td><td align="center">23.3&nbsp;/&nbsp;13.3k</td><td align="center">78.8&nbsp;/&nbsp;5.2k</td><td align="center">23.2&nbsp;/&nbsp;10.7k</td><td align="center">47.6&nbsp;/&nbsp;9.4k</td><td align="center">—</td></tr>
<tr><td align="left"><a href="https://huggingface.co/amao0o0/InfoDensity-DeepSeek-R1-Distill-Llama-8B"><b>+ InfoDensity</b></a></td><td align="center"><b>80.0&nbsp;/&nbsp;5.4k</b></td><td align="center"><b>40.0&nbsp;/&nbsp;10.0k</b></td><td align="center"><b>82.8&nbsp;/&nbsp;3.2k</b></td><td align="center"><b>30.8&nbsp;/&nbsp;7.6k</b></td><td align="center"><b>58.4&nbsp;/&nbsp;6.5k</b></td><td align="center"><b>+0.99</b></td></tr>
</tbody>
</table>

Cells are accuracy (%) / mean response tokens, over greedy pass@1 decoding; **+ InfoDensity** rows
link to the released weights. Overall is the mean over the four benchmarks. Raw per-benchmark
summaries are in [`results/`](results/).

**AES** (Accuracy–Efficiency Score) puts the trade-off in one number, measured against the base
model with $L$ the mean response length and $A$ the mean accuracy:

- $\Delta L = (L_{\text{base}} - L_{\text{model}}) \, / \, L_{\text{base}}$ &nbsp; relative length reduction
- $\Delta A = (A_{\text{model}} - A_{\text{base}}) \, / \, A_{\text{base}}$ &nbsp; relative accuracy change
- $\text{AES} = \alpha \, \Delta L + \beta \, \Delta A$ &nbsp; when $\Delta A \ge 0$, &nbsp; else $\alpha \, \Delta L - \gamma \, |\Delta A|$

with $\alpha = 1$, $\beta = 3$, $\gamma = 5$. A gain counts triple and a loss quintuple, so brevity
bought with accuracy cannot score well.

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

`--benchmark` accepts `amc23`, `aime24`, `math500` and `gpqa` — and `deepmath_val`, the in-domain validation split, once [`training/prepare_data.py`](training/prepare_data.py) has
built it. Once the four have run, score them:

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

[`training/`](training/) holds the reward and its data; the RL loop is
[verl](https://github.com/volcengine/verl). An incorrect trace scores 0, so brevity can never be
bought with accuracy. A correct one is scored on two factors:

- **$R_{\text{quality}}$** — the entropy vLLM already produced while sampling the trace, chunked
  into a trajectory and replaced by its suffix-max envelope $E_t = \max(H_t, \dots, H_T)$ so
  trailing filler cannot hide behind a confident prefix, then divided by $\log K$ so every trace
  shares one anchor.
- **$R_L$** — $\exp(-\lambda z)$, with $z$ the trace's length z-score among the correct traces for
  the same prompt.

One patch to verl is needed for that entropy to reach the reward; without it $R_{\text{quality}}$ is
silently skipped. [`training/README.md`](training/README.md) has it, plus the data prep and the
hydra overrides.

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
