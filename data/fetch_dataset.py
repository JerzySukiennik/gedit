"""Kaggle/CPU dataset builder for Gedit.

Streams `timbrooks/instructpix2pix-clip-filtered` from Hugging Face (it's far
too large to keep in full — see SPEC.md #2), takes the first `--n` pairs,
resizes both images to `--res` and writes them as a single raw uint8 binary
(mmap-friendly, same idea as MicroG's pl_train.bin) plus a separate float32
binary of frozen CLIP per-token text sequence embeddings for the edit
instructions (full sequence, not pooled — see model/clip_encoder.py for why).

Run once; the output is meant to become a Kaggle Dataset consumed by
train/train.py.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def build(args):
    images_path = f"{args.out_prefix}_images.bin"
    text_path = f"{args.out_prefix}_text.bin"
    meta_path = f"{args.out_prefix}_meta.json"
    prompts_path = f"{args.out_prefix}_prompts.json"

    sample_bytes = 2 * 3 * args.res * args.res
    if os.path.exists(images_path) and os.path.getsize(images_path) >= args.n * sample_bytes:
        print(f"{images_path} already has {args.n} samples — nothing to do")
        return

    from datasets import load_dataset
    ds = load_dataset("timbrooks/instructpix2pix-clip-filtered", split="train", streaming=True)

    prompts = []
    written = 0
    with open(images_path, "wb") as f_img:
        for row in ds:
            if written >= args.n:
                break
            try:
                before = row["original_image"].convert("RGB").resize(
                    (args.res, args.res), Image.LANCZOS)
                after = row["edited_image"].convert("RGB").resize(
                    (args.res, args.res), Image.LANCZOS)
            except Exception as e:
                # A handful of rows in this dataset have corrupt/missing
                # images — skip rather than crash a multi-hour prep run over
                # a few bad rows.
                print(f"skip malformed row (after {written} good pairs): {type(e).__name__}")
                continue
            before_arr = np.array(before, dtype=np.uint8).transpose(2, 0, 1)
            after_arr = np.array(after, dtype=np.uint8).transpose(2, 0, 1)
            f_img.write(before_arr.tobytes())
            f_img.write(after_arr.tobytes())
            prompts.append(row["edit_prompt"])
            written += 1
            if written % 1000 == 0:
                print(f"{written}/{args.n}", flush=True)

    print(f"images done: {written} pairs, {os.path.getsize(images_path)/1e9:.2f} GB")

    # --- text embeddings (frozen CLIP, see model/clip_encoder.py) -----------
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from model.clip_encoder import ClipTextEncoder

    encoder = ClipTextEncoder(device="cpu")
    emb = encoder.encode(prompts, batch_size=args.clip_batch)  # [N, seq_len, embed_dim]
    emb.numpy().astype(np.float32).tofile(text_path)
    print(f"text embeddings done: {tuple(emb.shape)}, {os.path.getsize(text_path)/1e6:.1f} MB")

    with open(meta_path, "w") as f:
        json.dump({
            "n": written,
            "val_n": min(args.val_n, written // 10),
            "res": args.res,
            "text_dim": encoder.embed_dim,
            "seq_len": encoder.seq_len,
            "clip_model": encoder.model.name_or_path,
            "prompts_sample": prompts[:20],
        }, f, indent=2)
    print(f"wrote {meta_path}")

    # Full prompt list, index-aligned with the images/text binaries — lets
    # runtime/sample_check.py show the actual instruction next to each
    # generated image instead of just judging structural reconstruction.
    with open(prompts_path, "w") as f:
        json.dump(prompts, f)
    print(f"wrote {prompts_path} ({len(prompts)} prompts)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--val-n", type=int, default=300)
    p.add_argument("--clip-batch", type=int, default=64)
    p.add_argument("--out-prefix", default="./gedit")
    build(p.parse_args())
