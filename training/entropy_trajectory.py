"""Entropy-trajectory quality term of the InfoDensity reward.

A reasoning trace is cut into fixed-size chunks of generated tokens. Each chunk
gets a mean per-token entropy, giving an entropy trajectory

    H = (H_1, ..., H_T),   H_t = mean per-token entropy of chunk t.

A trace is *information-dense* when its entropy falls early and stays low. Two
transformations turn that intuition into a bounded score:

1. **Suffix-max envelope.** Reading right to left,

       E_t = max(H_t, H_{t+1}, ..., H_T)

   so every chunk is charged with the highest entropy that still lies ahead of
   it. A trace that settles and then drifts back into high-entropy filler cannot
   hide that filler behind a low prefix — the late chunks propagate backward and
   raise the whole envelope. For a trace whose entropy decreases monotonically
   the envelope is the trajectory itself.

2. **Trace-external normalisation.** The envelope is divided by log K, the
   entropy of a uniform distribution over K candidate continuations. Unlike
   normalising by the trace's own first or maximum chunk, this anchor is the
   same constant for every trace in the batch, so a trace cannot improve its
   score by inflating its own reference point.

The quality term is then

    R_quality = clip(1 - mean_t(E_t) / log K, 0, 1)

which is 1 for a trace that resolves immediately and 0 for one that stays at or
above the uniform-entropy anchor throughout.
"""

import math
from typing import List, Sequence

__all__ = ["chunk_mean_entropies", "suffix_max_envelope", "quality_score"]


def chunk_mean_entropies(entropies: Sequence[float], chunk_size: int) -> List[float]:
    """Cut a per-token entropy sequence into chunks and average each one."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out = []
    for start in range(0, len(entropies), chunk_size):
        chunk = entropies[start : start + chunk_size]
        if chunk:
            out.append(sum(chunk) / len(chunk))
    return out


def suffix_max_envelope(trajectory: Sequence[float]) -> List[float]:
    """Right-to-left running maximum. Single O(T) pass."""
    env = list(trajectory)
    for i in range(len(env) - 2, -1, -1):
        if env[i] < env[i + 1]:
            env[i] = env[i + 1]
    return env


def quality_score(trajectory: Sequence[float], top_k: int, epsilon: float = 1e-6) -> float:
    """R_quality for one entropy trajectory.

    Args:
        trajectory: per-chunk mean entropies, in generation order.
        top_k: K in the log K normalisation anchor. Set it from your training
            config; it controls how wide a range R_quality spans.
        epsilon: numerical guard on the division.

    Returns:
        A score in [0, 1]; higher means the trace reached low entropy sooner and
        stayed there.
    """
    if len(trajectory) < 2:
        # Too short to have a trajectory; leave the base reward untouched.
        return 1.0
    if top_k <= 1:
        raise ValueError("top_k must be > 1 so that log K > 0")

    envelope = suffix_max_envelope(trajectory)
    auc = (sum(envelope) / len(envelope)) / (math.log(top_k) + epsilon)
    return max(0.0, min(1.0, 1.0 - auc))
