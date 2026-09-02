"""InfoDensity reward function for verl's batch reward manager.

    R_InfoDensity(tau) = R_quality(tau) * R_L(tau)   if tau reaches the correct answer
                       = 0                            otherwise

* ``R_quality`` scores how information-dense the trace is, from the suffix-max
  envelope of its per-token entropy trajectory (see ``entropy_trajectory.py``).
* ``R_L`` is a group-relative length scaling: within the set of *correct* traces
  sampled for the same prompt, a trace of length L gets ``exp(-lambda * z)``
  where z is L's z-score in that group. Shorter-than-typical correct traces are
  scaled up, longer ones down, and the group mean keeps the scale anchored so
  the signal does not collapse as the policy gets shorter overall.
* Both factors are applied only to correct traces. An incorrect trace scores 0
  regardless of how short or how dense it is, so the reward can never buy
  brevity with accuracy.

The entropy trajectory is the *rollout* entropy of the policy that generated the
trace — it needs no extra forward pass, but it does need verl to forward those
values to the reward function. See ``verl_integration.md``.

Register it with verl as::

    custom_reward_function.path=/path/to/training/infodensity_reward.py
    custom_reward_function.name=compute_score
    reward_model.reward_manager=batch

and pass ``length_coef``, ``entropy_chunk_size`` and ``entropy_top_k`` through
``custom_reward_function.reward_kwargs`` — they have no defaults here because
their values are a property of your training setup, not of the method.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from entropy_trajectory import chunk_mean_entropies, quality_score

try:
    from mathruler.grader import extract_boxed_content, grade_answer
except ImportError as exc:  # pragma: no cover
    raise ImportError("InfoDensity's correctness check needs mathruler: pip install mathruler") from exc


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_tokenizer_cache: Dict[str, Any] = {}


def _token_length(text: str, tokenizer_name: str) -> int:
    if tokenizer_name not in _tokenizer_cache:
        from transformers import AutoTokenizer

        _tokenizer_cache[tokenizer_name] = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
    return len(_tokenizer_cache[tokenizer_name].encode(text, add_special_tokens=False))


def _is_correct(response: str, ground_truth: str) -> bool:
    try:
        predicted = extract_boxed_content(response)
    except Exception:
        return False
    if predicted is None:
        return False
    try:
        return bool(grade_answer(predicted, ground_truth))
    except Exception:
        return str(predicted).strip() == str(ground_truth).strip()


def _thinking_entropies(response: str, entropies: List[float], tokenizer_name: str) -> List[float]:
    """Entropies of the deliberation span only, dropping the final answer.

    The quality term is about how the model *converges*, so the trajectory is cut
    at ``</think>`` when the trace has one. Traces without a think block are
    scored over their full length.
    """
    match = _THINK_BLOCK.search(response)
    if match is None:
        return list(entropies)
    n_tokens = _token_length(response[: match.end()], tokenizer_name)
    return list(entropies[:n_tokens])


def compute_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict[str, Any]],
    *,
    tokenizer_name: str,
    length_coef: float,
    entropy_chunk_size: int,
    entropy_top_k: int,
    correct_reward: float = 1.0,
    epsilon: float = 1e-6,
    log_frequency: int = 0,
    **kwargs,
) -> List[float]:
    """Score one rollout batch.

    Args:
        data_sources / solution_strs / ground_truths / extra_infos: the batch, as
            passed by verl's batch reward manager. ``extra_infos[i]`` must carry
            ``"index"`` (the prompt id, so the n rollouts of one prompt can be
            grouped) and, for the quality term, ``"entropys"`` (per-token rollout
            entropy).
        tokenizer_name: HF id of the policy's tokenizer, used for token counts.
        length_coef: lambda in ``R_L = exp(-lambda * z)``.
        entropy_chunk_size: C, the number of tokens per entropy-trajectory chunk.
        entropy_top_k: K in the log K normalisation anchor.
        correct_reward: base reward for a correct trace before both factors.
        epsilon: numerical guard.
        log_frequency: print batch statistics every N calls; 0 disables.

    Returns:
        One reward per sample, in the order given.
    """
    n = len(solution_strs)

    # --- correctness gate -------------------------------------------------
    rewards = [correct_reward if _is_correct(r, gt) else 0.0
               for r, gt in zip(solution_strs, ground_truths)]
    correct = [i for i in range(n) if rewards[i] > 0.0]

    # --- R_quality --------------------------------------------------------
    qualities: Dict[int, float] = {}
    for i in correct:
        entropies = extra_infos[i].get("entropys")
        if entropies is None:
            # Rollout entropy unavailable (e.g. the validation split, or verl not
            # yet patched). Leave the base reward alone rather than guessing.
            continue
        thinking = _thinking_entropies(solution_strs[i], entropies, tokenizer_name)
        if len(thinking) < entropy_chunk_size:
            continue
        trajectory = chunk_mean_entropies(thinking, entropy_chunk_size)
        if len(trajectory) < 2:
            continue
        q = quality_score(trajectory, entropy_top_k, epsilon)
        qualities[i] = q
        rewards[i] *= q

    # --- R_L: group-relative length scaling over correct traces -----------
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i in range(n):
        if "index" not in extra_infos[i]:
            raise KeyError(
                "extra_info['index'] is required to group the rollouts of one prompt; "
                "add it in your data preparation step (see prepare_data.py)."
            )
        groups[extra_infos[i]["index"]].append(i)

    lengths = {i: _token_length(solution_strs[i], tokenizer_name) for i in correct}
    for indices in groups.values():
        group_correct = [i for i in indices if rewards[i] > 0.0]
        if len(group_correct) <= 1:
            # A single correct trace has no group to be relative to.
            continue
        group_lengths = [lengths[i] for i in group_correct]
        mu = float(np.mean(group_lengths))
        sigma = float(np.std(group_lengths))
        for i in group_correct:
            z = (lengths[i] - mu) / (sigma + epsilon)
            rewards[i] *= float(np.exp(-length_coef * z))

    # --- optional logging --------------------------------------------------
    if log_frequency:
        compute_score.call_count = getattr(compute_score, "call_count", 0) + 1
        if compute_score.call_count % log_frequency == 0:
            arr = np.asarray(rewards)
            q_vals = list(qualities.values())
            q_note = (f"R_quality mean {np.mean(q_vals):.3f} over {len(q_vals)} traces"
                      if q_vals else "no entropy trajectories")
            print(
                f"[InfoDensity] batch {compute_score.call_count} | "
                f"correct {len(correct)}/{n} | "
                f"reward mean {arr.mean():.4f} max {arr.max():.4f} | {q_note}",
                flush=True,
            )

    return rewards
