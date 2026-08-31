"""Merge a LoRA adapter into its base model and save as a standalone HF dir.

The base model is read from the adapter's own adapter_config.json unless
--base_model is given explicitly.

Usage:
    python merge_lora.py \
        --lora_path      /path/to/checkpoint/lora_adapter \
        --tokenizer_dir  /path/to/checkpoint/huggingface \
        --output_dir     /path/to/merged_model
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=None,
                   help="HF id of the base model; default: read from adapter_config.json")
    p.add_argument("--lora_path", required=True, help="Dir with adapter_config.json + adapter_model.safetensors")
    p.add_argument("--tokenizer_dir", required=True, help="Dir with tokenizer.json/config.json from verl ckpt")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    if args.base_model is None:
        import json
        with open(Path(args.lora_path) / "adapter_config.json") as fh:
            args.base_model = json.load(fh)["base_model_name_or_path"]

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[merge] loading base {args.base_model} (dtype={args.dtype})")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, trust_remote_code=True)

    print(f"[merge] applying LoRA from {args.lora_path}")
    model = PeftModel.from_pretrained(base, args.lora_path)

    print("[merge] merge_and_unload()")
    merged = model.merge_and_unload()

    print(f"[merge] saving merged model to {out}")
    merged.save_pretrained(out, safe_serialization=True)

    # Copy tokenizer + config from the verl ckpt's huggingface/ dir so vllm can find them.
    src = Path(args.tokenizer_dir)
    for name in [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
        "generation_config.json",
    ]:
        s = src / name
        if s.exists():
            shutil.copy2(s, out / name)
            print(f"[merge] copied {name}")

    # Sanity: ensure config.json from base model is preserved (save_pretrained writes its own)
    print(f"[merge] done. listing {out}:")
    for f in sorted(out.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:40s} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
