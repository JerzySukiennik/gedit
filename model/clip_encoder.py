"""Frozen pretrained CLIP text encoder — the one part of Gedit NOT trained
from scratch (see SPEC.md #2: hand-training a text encoder good enough to
understand arbitrary edit instructions is a separate project the size of
MicroG itself). Converts a free-form edit instruction into a pooled
embedding, used to condition the U-Net via FiLM (model/unet.py).
"""

import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

MODEL_NAME = "openai/clip-vit-base-patch32"


class ClipTextEncoder:
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
        self.model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.model.config.projection_dim  # 512 for ViT-B/32

    @torch.no_grad()
    def encode(self, prompts, batch_size=64):
        """prompts: list[str] -> float32 tensor [N, embed_dim]."""
        out = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            tok = self.tokenizer(batch, padding=True, truncation=True,
                                  return_tensors="pt").to(self.device)
            emb = self.model(**tok).text_embeds
            out.append(emb.cpu())
        return torch.cat(out, dim=0).float()
