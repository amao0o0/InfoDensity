"""Build the DeepMath-103K training subset used in the paper.

Keeps the difficulty 5-10 slice of zwhe99/DeepMath-103K, takes the first
``--sample_size`` examples, and writes a 95/5 train/test split as verl-format
parquet.

Each row carries ``extra_info.index``, the prompt id. The InfoDensity reward
needs it to group the n rollouts sampled for one prompt when it computes the
group-relative length term.

Usage:
    python training/prepare_data.py --local_dir data/deepmath_subset
"""

import argparse
import os

from datasets import load_dataset

PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}"


def convert(split):
    def process_fn(example, idx):
        question = example.pop("question", "").strip()
        final_answer = example.pop("final_answer", "").strip()
        return {
            "data_source": "deepmath_subset",
            "prompt": [{"role": "user", "content": question + PROMPT_SUFFIX}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": final_answer},
            "extra_info": {"index": idx, "split": split},
        }

    return process_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", required=True, help="output dir for train.parquet / test.parquet")
    p.add_argument("--sample_size", type=int, default=7000, help="examples to keep after filtering")
    p.add_argument("--min_difficulty", type=int, default=5)
    p.add_argument("--max_difficulty", type=int, default=10)
    p.add_argument("--test_size", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds = load_dataset("zwhe99/DeepMath-103K")["train"]
    ds = ds.filter(lambda ex: args.min_difficulty <= ex["difficulty"] <= args.max_difficulty)
    if args.sample_size and args.sample_size < len(ds):
        ds = ds.select(range(args.sample_size))

    split = ds.train_test_split(test_size=args.test_size, seed=args.seed)
    train = split["train"].map(function=convert("train"), with_indices=True)
    test = split["test"].map(function=convert("test"), with_indices=True)

    os.makedirs(args.local_dir, exist_ok=True)
    train.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test.to_parquet(os.path.join(args.local_dir, "test.parquet"))
    print(f"wrote {args.local_dir}: train {len(train)} | test {len(test)}")


if __name__ == "__main__":
    main()
