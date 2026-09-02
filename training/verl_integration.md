# Forwarding rollout entropy to the reward function

InfoDensity's quality term reads the **per-token entropy of the trace as it was
generated**. vLLM already computes those values during rollout, so no extra
forward pass is needed — but stock verl drops them before the reward function
runs. One change to the batch reward manager keeps them.

Without it, `extra_infos[i]["entropys"]` is `None`, `R_quality` is skipped, and
what you are training is the length term alone.

## The change

In `verl/workers/reward_manager/batch.py`, inside `BatchRewardManager.verify()`,
the loop that fills `extras` also copies the rollout entropies and the response
token ids, both truncated to the sample's valid response length:

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

`valid_response_lengths` and `response_ids` are already local variables in
`verify()`; nothing else in the method changes. The guard on `entropys_batch`
keeps the manager working on batches that carry no entropies, such as the
validation split.

`entropys` is a plain `list[float]` by the time the reward function sees it, one
value per response token, aligned with `valid_response_ids`.

## Producing `data.batch["entropys"]`

The rollout worker only stores entropies when log-prob calculation is on:

```
actor_rollout_ref.rollout.calculate_log_probs=True
```

On recent verl (0.7.x) that is all that is required — `vllm_rollout_spmd.py`
computes `rollout_entropies` and puts them in the batch. On older versions you
may need to check that your rollout worker writes an `entropys` entry.

## Wiring the reward function

```
reward_model.reward_manager=batch
custom_reward_function.path=/path/to/training/infodensity_reward.py
custom_reward_function.name=compute_score
+custom_reward_function.reward_kwargs.tokenizer_name=<policy model id>
+custom_reward_function.reward_kwargs.length_coef=<lambda>
+custom_reward_function.reward_kwargs.entropy_chunk_size=<C>
+custom_reward_function.reward_kwargs.entropy_top_k=<K>
```

`infodensity_reward.py` imports `entropy_trajectory` as a top-level module, so
put `training/` on `PYTHONPATH`.

## Checking it works

The reward function is a pure function of a batch, so the integration can be
checked without a training run:

```python
import numpy as np
from infodensity_reward import compute_score

# One prompt, two correct traces of different lengths.
rewards = compute_score(
    data_sources=["deepmath_subset"] * 2,
    solution_strs=["<think>" + "reasoning " * 200 + "</think> The answer is \\boxed{4}",
                   "<think>" + "reasoning " * 600 + "</think> The answer is \\boxed{4}"],
    ground_truths=["4", "4"],
    extra_infos=[{"index": 0, "entropys": list(np.linspace(3.0, 0.1, 400))},
                 {"index": 0, "entropys": list(np.linspace(3.0, 0.1, 1200))}],
    tokenizer_name="<policy model id>",
    length_coef=...,      # your training values
    entropy_chunk_size=...,
    entropy_top_k=...,
)
print(rewards)  # shorter trace should score higher; both non-zero
```

If both rewards come back equal, `entropys` is not reaching the function.

---

`batch.py` is part of [verl](https://github.com/volcengine/verl), Apache-2.0.
The snippet above is quoted from it and modified; the surrounding file keeps its
original license and copyright.
