"""
Kaggle cell 2 of 2 — training.

Settings: Accelerator GPU T4 x2, Internet ON, Persistence "Variables and
Files". Inputs: the 'gedit-data' Dataset from 01-prep.py, plus — on any run
after the first — the previous run's output as 'gedit-ckpt'.

Same resumable design as MicroG: everything needed to continue lands in
/kaggle/working/run every CKPT_EVERY steps; add that as input to the next
session and it picks up mid-stride.
"""

import glob
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/gedit.git"
WORK = "/kaggle/working"
OUT = f"{WORK}/run"

# Cross-attention + zero-init, measured 2026-07-22: ~0.75s/step on T4x2
# (slightly slower than FiLM's 0.7s/step, as expected — more compute per
# step). First zero-init checkpoint (step 8000, ~4.3 epochs on 60k pairs)
# was visibly better than the pre-zero-init run (no longer pure static) but
# still far from FiLM's comparable-stage coherence, and not yet
# distinguishing between prompts — cross-attention has a harder thing to
# learn (attention patterns, not just a global tone shift) and needs more
# than 4.3 epochs. STEPS raised to 24000 (~12.9 epochs, in line with what
# worked for FiLM) to give it a real chance before judging the architecture
# itself. Still a ceiling, not a target — ckpt.pt is written every
# CKPT_EVERY steps regardless, safe to grab and stop early.
BATCH, ACCUM, STEPS, WARMUP = 32, 1, 24000, 200

if os.path.exists(f"{WORK}/gedit"):
    subprocess.run(["git", "-C", f"{WORK}/gedit", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/gedit"], check=True)
os.chdir(f"{WORK}/gedit")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)

# Kaggle's input mount depth isn't fixed (seen both /kaggle/input/<slug>/ and
# /kaggle/input/datasets/<owner>/<slug>/ in practice) — recursive search finds
# it regardless.
hits = glob.glob("/kaggle/input/**/gedit_images.bin", recursive=True)
if not hits:
    print("gedit_images.bin not found. /kaggle/input contains:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count("/") - 2
        if depth > 3:
            continue
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files)[:12]:
            print("  " * (depth + 1) + f)
    raise SystemExit("attach the gedit-data dataset, or wait for it to finish building")

data_prefix = hits[0].replace("_images.bin", "")
print(f"data: {data_prefix}")

os.makedirs(OUT, exist_ok=True)
hits_ckpt = sorted(glob.glob("/kaggle/input/**/ckpt.pt", recursive=True))
resume = []
if hits_ckpt:
    shutil.copy(hits_ckpt[0], f"{OUT}/ckpt.pt")
    resume = ["--resume"]
    print(f"resuming from {hits_ckpt[0]}")
else:
    print("starting from scratch")

cmd = [sys.executable, "train/train.py",
       "--data", data_prefix,
       "--out", OUT,
       "--batch-size", str(BATCH),
       "--grad-accum", str(ACCUM),
       "--max-steps", str(STEPS),
       "--warmup", str(WARMUP),
       "--eval-every", "200",
       "--ckpt-every", "200",
       "--log-every", "20"] + resume
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

print("\nsave this notebook's output as a Dataset ('gedit-ckpt') to continue "
      "in the next session, or download run/ckpt.pt if training finished.")
