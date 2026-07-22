"""
Kaggle cell 1 of 2 — build the image + text binaries.

Run this ONCE. Downloads a subset of timbrooks/instructpix2pix-clip-filtered
and packs it into gedit_images.bin / gedit_text.bin / gedit_meta.json under
/kaggle/working, which you then save as a Dataset and feed to the training
notebook. Done here rather than from home for the same reason as MicroG:
Kaggle's connection is fast, a domestic download of this dataset is not.

Settings: GPU off (CPU-only stage), Internet ON.
Expect roughly 1.5-2h for 60k pairs at 128px (scales ~linearly with N).

N=60000 (raised from the original 20000, 2026-07-22): 40000 training steps
on the original 20k pairs would have been ~65 epochs over the same data with
zero augmentation — real risk of memorizing the training set instead of
learning general edit patterns. The HF stream order is deterministic, so
this re-run's first 20000 pairs are exactly the old dataset plus 40000 new
ones, not a fresh unrelated sample — nothing already-downloaded is wasted in
spirit, though this script re-fetches from byte 0 rather than only the delta
(see the resume check below: it compares total file size to the new N, so a
smaller old file always triggers a full re-run).
"""

import os
import subprocess
import sys

REPO = "https://github.com/JerzySukiennik/gedit.git"
WORK = "/kaggle/working"
N, RES = 60000, 128

if os.path.exists(f"{WORK}/gedit"):
    # A stale checkout from an earlier attempt in this same session would
    # silently run old code even after this script itself was re-fetched —
    # pulling forces the checkout to match what curl just downloaded.
    subprocess.run(["git", "-C", f"{WORK}/gedit", "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/gedit"], check=True)
os.chdir(f"{WORK}/gedit")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "datasets", "transformers", "pillow"], check=True)

# A Hugging Face token lifts the anonymous rate limit. Put it in Kaggle's
# "Add-ons -> Secrets" as HF_TOKEN; without it this still works, just slower.
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF token loaded from Kaggle secrets")
except Exception as e:
    print(f"no HF token ({type(e).__name__}) — downloading anonymously, slower")

subprocess.run([sys.executable, "data/fetch_dataset.py",
                "--n", str(N), "--res", str(RES),
                "--out-prefix", f"{WORK}/gedit"], check=True)

print("\ndone — save this notebook's output as a Dataset named 'gedit-data'")
for f in sorted(os.listdir(WORK)):
    p = f"{WORK}/{f}"
    if os.path.isfile(p):
        print(f"  {f}  {os.path.getsize(p)/1e6:.1f} MB")
