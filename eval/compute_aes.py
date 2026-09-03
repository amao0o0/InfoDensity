"""Compute per-step and per-benchmark AES from eval/results_*/ directories.

AES (asymmetric, alpha=1 beta=3 gamma=5), as defined in the InfoDensity paper:
    if  ΔA >= 0:  AES = α·ΔL + β·ΔA
    if  ΔA <  0:  AES = α·ΔL - γ·|ΔA|
    ΔL = (L_base - L_model) / L_base   (relative length reduction)
    ΔA = (A_model - A_base) / A_base   (relative accuracy change)

Usage:
    python eval/compute_aes.py <results_dir>
    e.g. python eval/compute_aes.py results/infodensity_qwen3_8b
"""
import argparse
import glob
import json
import os
import sys

# Base-model reference numbers (paper Table 2, 'Original' rows). GPQA is scored with
# the multiple-choice fallback in eval_vllm.py, so these rows use that same rule.
BASE = {
    "Qwen/Qwen3-4B": {
        "amc23":   (90.0,  7600),
        "aime24":  (53.3, 11700),
        "math500": (90.6,  5000),
        "gpqa":    (43.9, 10400),
        "overall": (69.5,  8700),
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "amc23":   (80.0,  6600),
        "aime24":  (43.3, 11800),
        "math500": (85.0,  4200),
        "gpqa":    (24.2, 11300),
        "overall": (58.1,  8500),
    },
    "Qwen/Qwen3-8B": {
        "amc23":   (90.0,  8000),
        "aime24":  (63.3, 12200),
        "math500": (90.6,  5400),
        "gpqa":    (52.0,  9900),
        "overall": (74.0,  8900),
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "amc23":   (65.0,  8500),
        "aime24":  (23.3, 13300),
        "math500": (78.8,  5200),
        "gpqa":    (23.2, 10700),
        "overall": (47.6,  9400),
    },
}


def aes(a_model, l_model, a_base, l_base, alpha=1, beta=3, gamma=5):
    """Asymmetric AES: gain weighted by β=3, loss by γ=5."""
    if l_base == 0:
        return float("nan")
    dL = (l_base - l_model) / l_base
    dA = (a_model - a_base) / a_base if a_base else 0.0
    if dA >= 0:
        return alpha * dL + beta * dA
    return alpha * dL - gamma * abs(dA)


def infer_base(results_dir):
    """Guess the base model from the results dir name.

    Matched most-specific first: a name like ``infodensity_llama8b`` contains both
    "llama" and "8b", so order decides. Pass --base to skip the guessing.
    """
    name = os.path.basename(os.path.normpath(results_dir)).lower()
    if "llama" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    if "8b" in name:
        return "Qwen/Qwen3-8B"
    if "7b" in name or "dsr1" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    return "Qwen/Qwen3-4B"


def load_step(step_dir):
    out = {}
    for f in glob.glob(os.path.join(step_dir, "*.summary.json")):
        d = json.load(open(f))
        out[d["benchmark"]] = (d["accuracy"] * 100, d["avg_response_tokens"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", help="e.g. eval/results or eval/results_7b_4gpu")
    p.add_argument("--base", help="override base model id")
    args = p.parse_args()

    base_id = args.base or infer_base(args.results_dir)
    if base_id not in BASE:
        sys.exit(f"no reference numbers for base model {base_id!r}; "
                 f"known: {', '.join(sorted(BASE))}")
    base = BASE[base_id]
    print(f"# base model: {base_id}")
    print(f"# AES (asymmetric, α=1, β=3, γ=5)")
    print()

    step_dirs = sorted(glob.glob(os.path.join(args.results_dir, "step_*")), key=lambda x: int(x.rsplit("_", 1)[-1]))
    flat = not step_dirs  # a single run's summaries sitting directly in results_dir
    benches = ["amc23", "aime24", "math500", "gpqa"]
    hdr = ["step"] + [b for b in benches for _ in (0, 1)] + ["Acc", "Len", "AES"]
    cols = (4, 7, 4, 7, 4, 7, 4, 7, 4, 5, 5, 5)
    print(f"  {'step':>4}  ", end="")
    for b in benches:
        print(f"{b+'-Acc':>7s} {b+'-Len':>6s}  ", end="")
    print(f"{'Acc':>5s}  {'Len':>5s}  {'AES':>6s}")

    # baseline reference row
    aA, aL = base["amc23"]; iA, iL = base["aime24"]; mA, mL = base["math500"]; gA, gL = base["gpqa"]; oA, oL = base["overall"]
    print(f"  base   {aA:>5.1f} {aL/1000:>5.1f}k  {iA:>5.1f} {iL/1000:>5.1f}k  {mA:>5.1f} {mL/1000:>5.1f}k  {gA:>5.1f} {gL/1000:>5.1f}k  {oA:>5.1f}  {oL/1000:>4.1f}k  {0.000:>6.3f}")

    for sd in ([args.results_dir] if flat else step_dirs):
        label = "run" if flat else f"{int(os.path.basename(sd).rsplit('_', 1)[-1]):>4d}"
        r = load_step(sd)
        if len(r) < 4:
            missing = [b for b in benches if b not in r]
            print(f"  {label:>4}  incomplete (missing: {missing})")
            continue
        accs = [r[b][0] for b in benches]
        lens = [r[b][1] for b in benches]
        o_acc = sum(accs) / 4
        o_len = sum(lens) / 4
        bA, bL = base["overall"]
        score = aes(o_acc, o_len, bA, bL)
        row = f"  {label:>4}  "
        for a, l in zip(accs, lens):
            row += f"{a:>5.1f} {l/1000:>5.1f}k  "
        row += f"{o_acc:>5.1f}  {o_len/1000:>4.1f}k  {score:>+6.3f}"
        print(row)


if __name__ == "__main__":
    main()
