# Training

The reward and the data it trains on — not a turnkey cluster harness. The RL loop itself is
[verl](https://github.com/volcengine/verl); these pieces plug into it. Hyperparameters (λ, chunk
size C, top-K, and the usual PPO/GRPO knobs) are arguments here, not constants.

```
training/
├── infodensity_reward.py   the reward: correctness gate x R_quality x R_L
├── entropy_trajectory.py   suffix-max envelope and the R_quality term
├── prepare_data.py         DeepMath-103K subset -> verl parquet
└── verl_integration.md     forwarding rollout entropy to the reward function
```

## The reward

$$R_{\text{InfoDensity}}(\tau) = R_{\text{quality}}(\tau)\cdot R_{L}(\tau)\quad\text{if }\tau\text{ is correct},\qquad 0\ \text{otherwise.}$$

An incorrect trace scores 0 however short or dense it is, so neither factor can buy brevity with
accuracy.

**$R_{\text{quality}}$.** Per-token rollout entropy is cut into chunks of C tokens and averaged,
giving a trajectory $H_1 \dots H_T$ over the deliberation span, which is then replaced by its
suffix-max envelope $E_t = \max(H_t,\dots,H_T)$ — each chunk is charged with the highest entropy
still ahead of it, so a trace that resolves and then drifts back into filler cannot hide that filler
behind a confident prefix. Dividing by $\log K$, the entropy of a uniform choice among K
continuations, anchors every trace in the batch to the same constant; normalising by the trace's own
first or maximum chunk would instead let it inflate its own reference point.

$$R_{\text{quality}} = \mathrm{clip}\!\left(1 - \frac{1}{T}\sum_t \frac{E_t}{\log K},\ 0,\ 1\right)$$

**$R_L$.** Among the *correct* traces sampled for the same prompt, a trace of length $L_i$ is scaled
by $\exp(-\lambda z_i)$, with $z_i$ its z-score in that group. Comparing within a prompt avoids
holding easy and hard problems to one absolute budget, and the group mean re-centres as the policy
shortens, so the pressure does not fade once every trace is short.

Both factors read `extra_infos[i]`: `"index"` to group a prompt's rollouts (set by
`prepare_data.py`) and `"entropys"` for the trajectory.

## Running it

**1. Data.**

```bash
python training/prepare_data.py --local_dir data/deepmath_subset
```

The difficulty 5-10 slice of [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K),
7000 problems, split 95/5.

**2. Patch verl so the entropies reach the reward.** Two edits, one in
`verl/workers/reward_manager/batch.py` and one in `verl/trainer/ppo/ray_trainer.py`, against
**verl v0.6.1**, plus a non-zero `actor.entropy_coeff`. Without them `R_quality` is silently
skipped and you are training the length term alone.

**3. Point verl's `batch` reward manager at `compute_score`**, passing `tokenizer_name`,
`length_coef`, `entropy_chunk_size` and `entropy_top_k` through
`custom_reward_function.reward_kwargs`. [`verl_integration.md`](verl_integration.md) has the two
edits, a full run command, and the line to look for that tells you the entropies are arriving.

The released checkpoints were trained with LoRA; the reward is indifferent to that choice.

**4. Evaluate.** Merge the adapter and score the four benchmarks with the
pipeline in [`../eval/`](../eval):

```bash
python eval/merge_lora.py --lora_path <ckpt>/lora_adapter \
                          --tokenizer_dir <ckpt>/huggingface \
                          --output_dir <merged>
python eval/eval_vllm.py --model_path <merged> --benchmark math500 \
                         --temperature 0 --max_tokens 16384 --output_dir <out>
python eval/compute_aes.py <out> --base <base model id>
```

## Notes

- `R_quality` reads the entropy vLLM already produced while sampling the trace: no judge model, no
  second forward pass.
- When entropy is missing — the validation split, or traces shorter than one chunk — `R_quality` is
  skipped rather than guessed, and the trace keeps its base reward.
- Traces without a `<think>` block are scored over their full length.

## License

Apache-2.0, as the rest of this repository. `verl_integration.md` quotes and
modifies a file from [verl](https://github.com/volcengine/verl) (Apache-2.0);
`prepare_data.py` builds on the data-preparation script from
[ETR](https://github.com/Xuan1030/ETR) (Apache-2.0).
