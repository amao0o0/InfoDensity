"""4-cell rescue/regression analysis: Base vs InfoDensity.

Pools all 4 benchmarks (AMC23 + AIME24 + MATH500 + GPQA-D) into 4 cells:
  Preserved    (base_correct AND info_correct)
  Rescued      (base_wrong   AND info_correct)
  Still wrong  (base_wrong   AND info_wrong)
  Regression   (base_correct AND info_wrong)

For each cell: count, %, median lengths (base/info), Δmedian, %shorter, %base hits 16k cap.

This reads the per-sample .jsonl written by eval_vllm.py, not the .summary.json.
Those generations are far too large to ship, so results/ does not contain them --
run eval_vllm.py on the base model and on an InfoDensity model first, then point
this at the two output directories. --base_tag/--info_tag are the filename
prefixes eval_vllm.py derived from each --model_path.

Usage:
    python eval/four_cell_analysis.py \
        --base_dir  runs/base  --base_tag Qwen3-8B \
        --info_dir  runs/info  --info_tag InfoDensity-Qwen3-8B
"""
import argparse
import json
import statistics as st
from pathlib import Path


def load_jsonl(path):
    return {json.loads(l)["idx"]: json.loads(l) for l in open(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True,
                   help="Dir holding the base model's per-sample .jsonl")
    p.add_argument("--info_dir", required=True,
                   help="Dir holding the InfoDensity model's per-sample .jsonl")
    p.add_argument("--base_tag", required=True,
                   help="Filename prefix for the base run, e.g. 'Qwen3-8B'")
    p.add_argument("--info_tag", required=True,
                   help="Filename prefix for the InfoDensity run, e.g. 'InfoDensity-Qwen3-8B'")
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    info_dir = Path(args.info_dir)

    benches = ["amc23", "math500", "aime24", "gpqa"]

    cells = {(True, True): [], (True, False): [], (False, True): [], (False, False): []}
    per_bench = {b: {k: 0 for k in cells} for b in benches}

    missing = []
    for b in benches:
        bf = base_dir / f"{args.base_tag}__{b}__t0.0.jsonl"
        infof = info_dir / f"{args.info_tag}__{b}__t0.0.jsonl"
        if not bf.exists():
            missing.append(str(bf)); continue
        if not infof.exists():
            missing.append(str(infof)); continue
        base = load_jsonl(bf)
        info = load_jsonl(infof)
        # Align by idx (intersection)
        common = sorted(set(base) & set(info))
        for idx in common:
            bj, ij = base[idx], info[idx]
            key = (bj["correct"], ij["correct"])
            cells[key].append((bj["response_tok_len"], ij["response_tok_len"]))
            per_bench[b][key] += 1

    if missing:
        print("ERROR: missing files:")
        for m in missing: print(f"  {m}")
        return

    label = {(True, True):  "Preserved   (✓→✓)",
             (False, True): "Rescued     (✗→✓)",
             (False, False): "Still wrong (✗→✗)",
             (True, False): "Regression  (✓→✗)"}

    n_total = sum(len(v) for v in cells.values())
    print(f"Pooled n={n_total} across {len(benches)} benchmarks")
    print(f"  Base : {args.base_dir}/{args.base_tag}__*")
    print(f"  Info : {args.info_dir}/{args.info_tag}__*")
    print()
    print(f"{'Cell':22s}  {'n':>4s}  {'%':>5s}  {'base_med':>9s}  {'info_med':>9s}  "
          f"{'Δmed':>7s}  {'%shorter':>9s}  {'%Base_cap':>10s}")
    for k in [(True, True), (False, True), (False, False), (True, False)]:
        rows = cells[k]
        n = len(rows)
        if n == 0:
            print(f"{label[k]:22s}  {n:>4d}  {0:>4.1f}%  {'-':>9s}  {'-':>9s}  {'-':>7s}  {'-':>8s}  {'-':>9s}")
            continue
        bls = [b for b, _ in rows]; ils = [i for _, i in rows]
        bm = st.median(bls); im = st.median(ils)
        pct_short = sum(1 for b, i in rows if i < b) / n * 100
        pct_bcap = sum(1 for b, _ in rows if b >= 16384) / n * 100
        print(f"{label[k]:22s}  {n:>4d}  {n/n_total*100:>4.1f}%  "
              f"{bm:>9.0f}  {im:>9.0f}  {bm-im:>+7.0f}  {pct_short:>8.0f}%  {pct_bcap:>9.0f}%")

    print()
    print("=== Per-benchmark cell counts ===")
    print(f"{'bench':<10s}  {'n':>4s}  " + "  ".join(f"{label[k][:14]:>14s}" for k in [(True,True),(False,True),(False,False),(True,False)]))
    for b in benches:
        n = sum(per_bench[b].values())
        cells_str = "  ".join(f"{per_bench[b][k]:>14d}" for k in [(True,True),(False,True),(False,False),(True,False)])
        print(f"{b:<10s}  {n:>4d}  {cells_str}")


if __name__ == "__main__":
    main()
