"""Recompute just the text embeddings for an existing packed dataset, after
an architecture change to model/clip_encoder.py (pooled -> full sequence,
2026-07-22, for cross-attention object-edit support).

Images/prompts already fetched by data/fetch_dataset.py don't need
re-downloading — only <prefix>_text.bin and <prefix>_meta.json change here.
Saves re-running the multi-hour HF stream download a second time.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.clip_encoder import ClipTextEncoder


def build(args):
    meta_path = f"{args.out_prefix}_meta.json"
    prompts_path = f"{args.out_prefix}_prompts.json"
    text_path = f"{args.out_prefix}_text.bin"

    with open(meta_path) as f:
        meta = json.load(f)
    with open(prompts_path) as f:
        prompts = json.load(f)
    if len(prompts) != meta["n"]:
        raise SystemExit(f"{prompts_path} has {len(prompts)} prompts but meta says n={meta['n']}")

    encoder = ClipTextEncoder(device="cpu")
    emb = encoder.encode(prompts, batch_size=args.clip_batch)  # [N, seq_len, embed_dim]
    emb.numpy().astype(np.float32).tofile(text_path)
    print(f"text embeddings done: {tuple(emb.shape)}, {os.path.getsize(text_path)/1e6:.1f} MB")

    meta["text_dim"] = encoder.embed_dim
    meta["seq_len"] = encoder.seq_len
    meta["clip_model"] = encoder.model.name_or_path
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"updated {meta_path}: text_dim={meta['text_dim']}, seq_len={meta['seq_len']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", default="./gedit", help="same prefix used by fetch_dataset.py")
    p.add_argument("--clip-batch", type=int, default=64)
    build(p.parse_args())
