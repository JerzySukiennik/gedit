"""Quick visual sanity check for a training checkpoint: DDIM-samples a
handful of validation pairs and saves a before | real-after | generated-after
grid PNG. This is how a checkpoint gets judged — SPEC.md #1 deliberately has
no automatic quality metric, Jurek looks at the pixels.

Needs only the packed data (data/fetch_dataset.py's output) and a
train/train.py checkpoint — no network, no transformers/CLIP needed, since
the validation examples already carry precomputed text embeddings.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.unet import UNet
from model.scheduler import DiffusionSchedule


def to_img(t):
    """[-1,1] CHW float tensor -> HWC uint8 numpy."""
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    return arr.transpose(1, 2, 0)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(f"{args.data}_meta.json") as f:
        meta = json.load(f)
    res, text_dim, n, val_n = meta["res"], meta["text_dim"], meta["n"], meta["val_n"]

    images = np.memmap(f"{args.data}_images.bin", dtype=np.uint8, mode="r",
                        shape=(n, 2, 3, res, res))
    text = np.memmap(f"{args.data}_text.bin", dtype=np.float32, mode="r",
                      shape=(n, text_dim))

    idx = list(range(n - val_n, n))[:args.n]

    prompts_path = f"{args.data}_prompts.json"
    prompts = None
    if os.path.exists(prompts_path):
        with open(prompts_path) as f:
            all_prompts = json.load(f)
        prompts = [all_prompts[i] for i in idx]
    else:
        print(f"no {prompts_path} — older data prep, showing images without instruction text")

    model = UNet(text_dim=text_dim).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded checkpoint at step {ckpt['step']}")

    schedule = DiffusionSchedule(device=device)

    before = torch.from_numpy(images[idx, 0].copy()).float().to(device) / 127.5 - 1.0
    after_real = torch.from_numpy(images[idx, 1].copy()).float().to(device) / 127.5 - 1.0
    text_emb = torch.from_numpy(text[idx].copy()).to(device)

    with torch.no_grad():
        generated = schedule.ddim_sample(model, before, text_emb, steps=args.steps, device=device)

    rows = [np.concatenate([to_img(before[i]), to_img(after_real[i]), to_img(generated[i])], axis=1)
            for i in range(len(idx))]
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"saved {args.out}  (columns: before | real after | generated, {len(idx)} rows)")
    if prompts:
        for i, p in enumerate(prompts):
            print(f"  row {i}: {p!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="prefix used by fetch_dataset.py's --out-prefix")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="./sample_check.png")
    main(p.parse_args())
