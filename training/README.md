# Training

The InfoDensity reward, and what it takes to train with it.

This is the reward and its data, not a turnkey cluster harness: the RL loop
itself is [verl](https://github.com/volcengine/verl), and the pieces here plug
into it. Hyperparameters (lambda, chunk size C, top-K, and the usual PPO/GRPO
knobs) are yours to set — they are arguments here, not constants.

```
training/
├── infodensity_reward.py   the reward: correctness gate x R_quality x R_L
├── entropy_trajectory.py   suffix-max envelope and the R_quality term
├── prepare_data.py         DeepMath-103K subset -> verl parquet
└── verl_integration.md     forwarding rollout entropy to the reward function
```

## The reward

$$R_{\text{InfoDensity}}(\tau) = R_{\text{quality}}(\tau)\cdot R_{L}(\tau)\quad\text{if }\tau\text{ is correct},\qquad 0\ \text{otherwise.}$$

**The correctness gate comes first.** An incorrect trace scores 0 no matter how
short or how dense it is, so neither factor can buy brevity with accuracy.

**$R_{\text{quality}}$ — how information-dense the trace is.** The trace's
per-token rollout entropy is cut into chunks of C tokens and averaged per chunk,
giving a trajectory $H_1 \dots H_T$ over the deliberation span. Two steps turn
it into a score:

- *Suffix-max envelope*: $E_t = \max(H_t,\dots,H_T)$, so each chunk is charged
  with the highest entropy still ahead of it. A trace that resolves and then
  drifts back into high-entropy filler cannot hide the filler behind a confident
  prefix — late chunks propagate backward and lift the whole envelope.
- *Trace-external anchor*: divide by $\log K$, the entropy of a uniform choice
  among K continuations. The same constant for every trace in the batch, so a
  trace cannot improve its score by inflating its own reference point — which is
  what normalising by the trace's own first or maximum chunk would allow.

$$R_{\text{quality}} = \mathrm{clip}\!\left(1 - \frac{1}{T}\sum_t \frac{E_t}{\log K},\ 0,\ 1\right)$$

**$R_L$ — group-relative length scaling.** Among the *correct* traces sampled for
the same prompt, a trace of length $L_i$ is scaled by $\exp(-\lambda z_i)$, where
$z_i$ is $L_i$'s z-score within that group. The comparison is within-prompt, so
easy and hard problems are not held to one absolute length budget, and the group
mean re-centres as the policy shortens — the pressure does not fade away once
every trace is short.

Both factors need `extra_infos[i]`: `"index"` to group a prompt's rollouts (set
by `prepare_data.py`) and `"entropys"` for the trajectory (see below).

## Running it

**1. Data.**

```bash
python training/prepare_data.py --local_dir data/deepmath_subset
```

The difficulty 5-10 slice of [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K),
7000 problems, split 95/5.

**2. Patch verl so rollout entropy reaches the reward.** One block in
`verl/workers/reward_manager/batch.py`, plus
`actor_rollout_ref.rollout.calculate_log_probs=True`. Without it `R_quality` is
silently skipped and you are training the length term alone —
[`verl_integration.md`](verl_integration.md) has the change and a check that
tells you whether it took.

**3. Point verl's `batch` reward manager at `compute_score`**, passing
`tokenizer_name`, `length_coef`, `entropy_chunk_size` and `entropy_top_k`
through `custom_reward_function.reward_kwargs`. The exact hydra overrides are in
[`verl_integration.md`](verl_integration.md).

The released checkpoints were trained with LoRA on top of each base model; the
reward is indifferent to that choice.

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

- **Entropy is free here.** `R_quality` reads the entropy vLLM already produced
  while sampling the trace. No judge model, no second forward pass.
- **`R_quality` is skipped, not guessed, when entropy is missing** — on the
  validation split, or on traces shorter than one chunk. Those traces keep their
  base reward.
- **Traces without a `<think>` block** are scored over their full length.

## License

Apache-2.0, as the rest of this repository. `verl_integration.md` quotes and
modifies a file from [verl](https://github.com/volcengine/verl) (Apache-2.0);
`prepare_data.py` builds on the data-preparation script from
[ETR](https://github.com/Xuan1030/ETR) (Apache-2.0).
