# Running InfoDensity on verl

`R_quality` reads the **per-token entropy of the trace as it was generated**. verl computes those
values, but discards them before the reward function runs, so two changes are needed. Without them
`extra_infos[i]["entropys"]` is `None`, `R_quality` is skipped, and what trains is the length term
alone — silently, with no error.

## Which verl

Pin **v0.6.1**:

```bash
git clone --branch v0.6.1 https://github.com/volcengine/verl.git
```

This matters. v0.7.0 removed the synchronous rollout path; its remaining path computes rewards
through a separate registry that has no `batch` manager, so the change below lands in a file that
is never called and training fails with `Unknown reward manager: batch`. v0.5.0 and earlier do not
have the code the change applies to.

## Change 1 — forward the entropies to the reward function

In `verl/workers/reward_manager/batch.py`, inside `BatchRewardManager.verify()`, the loop that
fills `extras` also copies the entropies and the response token ids, both truncated to the sample's
valid response length:

```python
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])

        entropys_batch = data.batch.get("entropys", None)                       # added
        for i in range(len(data)):
            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]
            if entropys_batch is not None:                                      # added
                valid_len = valid_response_lengths[i].item()                    # added
                extras[i]["entropys"] = entropys_batch[i][:valid_len].tolist()  # added
                extras[i]["valid_response_ids"] = response_ids[i][:valid_len].tolist()  # added
```

`valid_response_lengths` and `response_ids` are already local variables in `verify()`. The guard
keeps the manager working on batches that carry no entropies, such as the validation split.

## Change 2 — keep the entropies until the reward has run

In `verl/trainer/ppo/ray_trainer.py`, the entropies are produced by the `old_log_prob` step and
dropped before they reach the batch, and the reward is computed before that step runs at all. Two
edits fix the ordering:

1. Delete `old_log_prob.batch.pop("entropys")` — the line immediately above
   `batch = batch.union(old_log_prob)`.
2. Move the whole `with marked_timer("reward", ...):` block from above the "Operating Mode
   Selection" comment to just after that `union`, i.e. immediately before
   `assert "old_log_probs" in batch.batch`, and drop the entropies once it has run:

```python
                    with marked_timer("reward", timing_raw, color="yellow"):
                        ...                       # the block, moved verbatim
                    if "entropys" in batch.batch:
                        batch.batch.pop("entropys")
```

## Enable the entropy computation

The entropies exist only when the actor is asked for them, which on v0.6.1 is tied to the entropy
coefficient. **`actor.entropy_coeff` must be non-zero**, or `entropys` is never computed and the
reward silently falls back to the length term.

## A run

```bash
PYTHONPATH=/path/to/verl:/path/to/InfoDensity/training \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/deepmath_subset/train.parquet \
    data.val_files=data/deepmath_subset/test.parquet \
    actor_rollout_ref.model.path=<policy model id> \
    actor_rollout_ref.actor.entropy_coeff=<non-zero> \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward_model.reward_manager=batch \
    custom_reward_function.path=/path/to/InfoDensity/training/infodensity_reward.py \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.tokenizer_name=<policy model id> \
    +custom_reward_function.reward_kwargs.length_coef=<lambda> \
    +custom_reward_function.reward_kwargs.entropy_chunk_size=<C> \
    +custom_reward_function.reward_kwargs.entropy_top_k=<K> \
    +custom_reward_function.reward_kwargs.log_frequency=1 \
    trainer.n_gpus_per_node=<N> trainer.nnodes=1
```

Batch sizes, learning rate, rollout count and response length are yours to set.
`tensor_model_parallel_size` must divide the number of GPUs.

## Checking that the entropies arrive

`log_frequency=1` makes the reward print one line per batch:

```
[InfoDensity] batch 7 | correct 5/16 | reward mean 0.3120 max 0.8400 | R_quality mean 0.412 over 5 traces
```

`R_quality mean ... over N traces` with N > 0 is the confirmation. If it reads
`no entropy trajectories` while `correct` is non-zero, the entropies are not arriving and one of
the two changes above has not taken. On the validation split that message is expected — no
entropies are computed there, and those traces keep their base reward.

---

`batch.py` and `ray_trainer.py` are part of [verl](https://github.com/volcengine/verl), Apache-2.0.
The snippets above are quoted from them and modified; the files keep their original license.
