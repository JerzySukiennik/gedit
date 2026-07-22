"""Manual hands-on test: edit ANY photo with ANY free-form instruction using
a trained checkpoint, from the terminal — no gzowo-ai wiring needed yet.

This is the same model/scheduler.py + model/clip_encoder.py pipeline that
will eventually run inside gzowo-ai's Node bridge (via ONNX), just driven
directly in Python so Jurek can try real photos/prompts before that
integration is built (see SPEC.md "Kolejne kroki" #4-6).
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.unet import UNet
from model.scheduler import DiffusionSchedule
from model.clip_encoder import ClipTextEncoder


def load_image(path, res):
    img = Image.open(path).convert("RGB").resize((res, res), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    return torch.from_numpy(arr).float() / 127.5 - 1.0


def to_img(t):
    arr = ((t.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
    return arr.transpose(1, 2, 0)


def main(args):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet().to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded checkpoint at step {ckpt['step']}")

    print("loading CLIP text encoder (frozen, first run downloads it)...")
    encoder = ClipTextEncoder(device=device)
    text_seq = encoder.encode([args.prompt])

    before = load_image(args.image, args.res).unsqueeze(0).to(device)
    text_seq = text_seq.to(device)

    schedule = DiffusionSchedule(device=device)
    print(f"sampling ({args.steps} DDIM steps)...")
    with torch.no_grad():
        generated = schedule.ddim_sample(model, before, text_seq, steps=args.steps, device=device)

    grid = np.concatenate([to_img(before[0]), to_img(generated[0])], axis=1)
    Image.fromarray(grid).save(args.out)
    print(f"saved {args.out}  (left: input at {args.res}px | right: edited)")
    print(f"prompt: {args.prompt!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to any photo")
    p.add_argument("--prompt", required=True, help="free-form edit instruction")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--res", type=int, default=128, help="must match the resolution the checkpoint was trained at")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--out", default="./edited.png")
    main(p.parse_args())
