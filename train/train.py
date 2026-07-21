"""Gedit training loop — diffusion noise-prediction loss on (before, after,
text_emb) triples prepared by data/fetch_dataset.py. Mirrors MicroG's
train/train.py conventions: gradient accumulation, DataParallel across
Kaggle's T4x2, checkpointing every --ckpt-every steps so a killed 12h session
resumes clean.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Running this as `python train/train.py` puts train/ on sys.path, not the
# repo root — the same fix data/fetch_dataset.py already needed for its
# `from model.clip_encoder import ...` import.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.unet import UNet
from model.scheduler import DiffusionSchedule


class PairDataset(Dataset):
    def __init__(self, data_prefix, split="train"):
        with open(f"{data_prefix}_meta.json") as f:
            meta = json.load(f)
        self.res = meta["res"]
        self.text_dim = meta["text_dim"]
        n, val_n = meta["n"], meta["val_n"]
        self.images = np.memmap(f"{data_prefix}_images.bin", dtype=np.uint8, mode="r",
                                 shape=(n, 2, 3, self.res, self.res))
        self.text = np.memmap(f"{data_prefix}_text.bin", dtype=np.float32, mode="r",
                               shape=(n, self.text_dim))
        self.idx = range(0, n - val_n) if split == "train" else range(n - val_n, n)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        before = torch.from_numpy(self.images[j, 0].copy()).float() / 127.5 - 1.0
        after = torch.from_numpy(self.images[j, 1].copy()).float() / 127.5 - 1.0
        text = torch.from_numpy(self.text[j].copy())
        return before, after, text


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = PairDataset(args.data, "train")
    val_ds = PairDataset(args.data, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNet(text_dim=train_ds.text_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params/1e6:.1f}M")

    if torch.cuda.device_count() > 1 and not args.single_gpu:
        model = torch.nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")

    schedule = DiffusionSchedule(device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = f"{args.out}/ckpt.pt"
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print(f"resumed from step {step}")

    def cycle(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = cycle(train_loader)
    t0 = time.time()

    while step < args.max_steps:
        opt.zero_grad()
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            before, after, text_emb = next(data_iter)
            before, after, text_emb = before.to(device), after.to(device), text_emb.to(device)
            b = before.shape[0]
            t = torch.randint(0, schedule.timesteps, (b,), device=device)
            noisy_after, noise = schedule.q_sample(after, t)
            pred = model(torch.cat([noisy_after, before], dim=1), t, text_emb)
            loss = F.mse_loss(pred, noise) / args.grad_accum
            loss.backward()
            loss_accum += loss.item()

        if step < args.warmup:
            for g in opt.param_groups:
                g["lr"] = args.lr * (step + 1) / args.warmup
        opt.step()
        step += 1

        if step % args.log_every == 0:
            print(f"step {step}/{args.max_steps}  loss {loss_accum:.4f}  "
                  f"{time.time()-t0:.0f}s elapsed", flush=True)

        if step % args.eval_every == 0:
            model.eval()
            vloss, n_batches = 0.0, 0
            with torch.no_grad():
                for before, after, text_emb in val_loader:
                    before, after, text_emb = before.to(device), after.to(device), text_emb.to(device)
                    b = before.shape[0]
                    t = torch.randint(0, schedule.timesteps, (b,), device=device)
                    noisy_after, noise = schedule.q_sample(after, t)
                    pred = model(torch.cat([noisy_after, before], dim=1), t, text_emb)
                    vloss += F.mse_loss(pred, noise).item()
                    n_batches += 1
            print(f"  val loss {vloss / max(n_batches, 1):.4f}")
            model.train()

        if step % args.ckpt_every == 0 or step == args.max_steps:
            raw_model = model.module if hasattr(model, "module") else model
            torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(), "step": step},
                       ckpt_path)

    print(f"done — {args.max_steps} steps in {(time.time()-t0)/3600:.1f}h")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="prefix used by fetch_dataset.py's --out-prefix")
    p.add_argument("--out", default="./run")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=8000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--single-gpu", action="store_true")
    main(p.parse_args())
