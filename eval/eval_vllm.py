"""Offline-vLLM evaluation for math and science reasoning benchmarks.

Rewrite of the evaluation script from ETR (https://github.com/Xuan1030/ETR, Apache-2.0):
no SGLang server, vLLM runs in-process and scoring happens locally.

Reuses grader.py + math_normalize.py for answer equivalence.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Make grader / math_normalize importable as siblings.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from datasets import load_dataset
from grader import grade_answer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}"

# Benchmarks whose answers are option letters rather than free-form expressions.
MULTIPLE_CHOICE = {"gpqa"}


# --- boxed-answer extraction (verbatim from eval.py) -------------------------

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left): -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx: right_brace_idx + 1]


def extract_solution(solution_str):
    boxed = last_boxed_only_string(solution_str)
    if boxed is None:
        return None
    try:
        return remove_boxed(boxed)
    except AssertionError:
        return None


# --- multiple-choice fallback ------------------------------------------------

# On multiple-choice benchmarks a model may state the letter in the conventional
# form ("ANSWER: C", "Answer: B. <restated option>") rather than boxing it. This
# runs only when no boxed answer is present, so it never overrides an explicit one.
_CHOICE_PATTERNS = [
    (r"ANSWER:\s*\(?([ABCD])\)?\b", 0),
    (r"(?:final\s+)?answer\s+is\s*\**\s*\(?([ABCD])\)?\b", re.IGNORECASE),
]


def extract_choice(solution_str):
    """Recover a multiple-choice letter stated without \boxed{}. None if absent."""
    for pattern, flags in _CHOICE_PATTERNS:
        found = re.findall(pattern, solution_str, flags)
        if found:
            return found[-1].upper()
    return None


# --- benchmark loaders -------------------------------------------------------

def load_benchmark(name):
    """Returns a list of {question, answer} dicts."""
    if name == "amc23":
        ds = load_dataset("math-ai/amc23", split="test")
        # amc23 already has 'question' and 'answer'
        return [{"question": ex["question"], "answer": str(ex["answer"])} for ex in ds]
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"question": ex["problem"], "answer": str(ex["answer"])} for ex in ds]
    if name == "aime24":
        ds = load_dataset("math-ai/aime24", split="test")
        # math-ai/aime24 schema: id, problem, solution (often "\boxed{N}"), url.
        out = []
        for ex in ds:
            sol = ex["solution"]
            b = last_boxed_only_string(sol)
            ans = remove_boxed(b) if b else sol
            out.append({"question": ex["problem"], "answer": str(ans).strip()})
        return out
    if name == "gpqa":
        try:
            ds = load_dataset("fingertap/GPQA-Diamond", split="test")
        except Exception as e:
            raise RuntimeError(
                f"fingertap/GPQA-Diamond load failed ({e}); with an HF token set, "
                f"`Idavidrein/gpqa` carries the same questions."
            )
        # gpqa may have different field names; try common ones
        out = []
        for ex in ds:
            q = ex.get("question") or ex.get("problem") or ex.get("Question")
            a = ex.get("answer") or ex.get("Correct Answer") or ex.get("solution")
            if q is None or a is None:
                raise RuntimeError(f"unrecognized gpqa schema: {list(ex.keys())}")
            out.append({"question": q, "answer": str(a)})
        return out
    raise ValueError(f"unknown benchmark: {name}")


# --- main eval loop ----------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="HF model name or local path")
    p.add_argument("--benchmark", required=True,
                   choices=["amc23", "math500", "aime24", "gpqa"])
    p.add_argument("--limit", type=int, default=None,
                   help="Cap benchmark to first N items (smoke test)")
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--tp", type=int, default=1,
                   help="tensor_parallel_size for vllm")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default=None,
                   help="Where to write results JSON; default: ./outputs")
    p.add_argument("--trust_remote_code", action="store_true")
    args = p.parse_args()

    out_root = Path(args.output_dir) if args.output_dir else _HERE / "outputs"
    out_root.mkdir(parents=True, exist_ok=True)
    tag = (Path(args.model_path).name or "model").replace("/", "_")
    out_file = out_root / f"{tag}__{args.benchmark}__t{args.temperature}.jsonl"
    summary_file = out_root / f"{tag}__{args.benchmark}__t{args.temperature}.summary.json"

    print(f"[eval_vllm] model={args.model_path} benchmark={args.benchmark} "
          f"limit={args.limit} max_tokens={args.max_tokens} tp={args.tp}")

    # 1) load benchmark
    data = load_benchmark(args.benchmark)
    if args.limit:
        data = data[: args.limit]
    print(f"[eval_vllm] loaded {len(data)} problems")

    # 2) tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )

    prompts = []
    for ex in data:
        chat = [{"role": "user", "content": f"{ex['question']}\n\n{SYSTEM_PROMPT}"}]
        prompt = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False
        )
        prompts.append(prompt)

    # 3) vllm offline generation
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    sp = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    gen_secs = time.time() - t0
    print(f"[eval_vllm] generation done in {gen_secs:.1f}s")

    # 4) score
    num_correct = 0
    total_tok = 0
    records = []
    with open(out_file, "w") as fh:
        for i, (ex, out) in enumerate(zip(data, outputs)):
            response = out.outputs[0].text
            tok_len = len(out.outputs[0].token_ids)
            gt = ex["answer"]
            pred = extract_solution(response)
            if pred is None and args.benchmark in MULTIPLE_CHOICE:
                pred = extract_choice(response)
            try:
                ok = bool(grade_answer(pred, gt)) if pred is not None else False
            except Exception:
                ok = False
            num_correct += int(ok)
            total_tok += tok_len
            rec = {
                "idx": i,
                "question": ex["question"],
                "ground_truth": gt,
                "extracted": pred,
                "correct": ok,
                "response_tok_len": tok_len,
                "response": response,
            }
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            running = num_correct / (i + 1)
            print(f"[{i+1}/{len(data)}] gt={gt!s:<20} pred={str(pred)[:40]:<40} "
                  f"ok={ok} acc={running:.4f} tok={tok_len}")

    acc = num_correct / len(data)
    avg_len = total_tok / len(data)
    summary = {
        "model_path": args.model_path,
        "benchmark": args.benchmark,
        "n_problems": len(data),
        "num_correct": num_correct,
        "accuracy": acc,
        "avg_response_tokens": avg_len,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "tp": args.tp,
        "generation_seconds": gen_secs,
        # basename only: a summary is meant to be shareable, and the full path is
        # a property of the machine that produced it
        "results_file": out_file.name,
    }
    with open(summary_file, "w") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print(f"=== {args.benchmark} on {args.model_path} ===")
    print(f"accuracy:      {acc:.4f} ({num_correct}/{len(data)})")
    print(f"avg tok len:   {avg_len:.1f}")
    print(f"results:       {out_file}")
    print(f"summary:       {summary_file}")


if __name__ == "__main__":
    main()
