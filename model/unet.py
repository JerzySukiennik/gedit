"""Small conditional U-Net — the part of Gedit that IS trained from scratch.
Predicts the noise added to `after`, conditioned on `before` (concatenated on
the channel axis — same img2img formulation InstructPix2Pix itself uses) and
a pooled CLIP text embedding.

Deliberate simplification vs. real Stable-Diffusion-style U-Nets: FiLM
conditioning (scale/shift from a single pooled vector) instead of
cross-attention to a full token sequence. Cheaper, and enough for this
model's scope (style/filter edits + simple inpainting, see SPEC.md #1) —
cross-attention buys precision this scale of model and dataset can't use yet.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    """Sinusoidal embedding, same construction as the original DDPM paper."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLMResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, twice, with FiLM (scale, shift) from `cond`
    applied after the first norm. 1x1 conv on the residual path when the
    channel count changes.
    """

    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Plain multi-head self-attention over flattened spatial positions.
    Only used at the bottleneck — cheapest resolution, biggest payoff for
    long-range mixing on a small model.
    """

    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv.unbind(1)
        attn = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(c // self.heads), dim=-1)
        out = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """base_channels=64, channel_mults=(1,2,4) -> 64/128/256, 2 ResBlocks per
    level, self-attention only at the 4x bottleneck. Measured 12.9M params —
    deliberately far below MicroG's 110M: images at 64-128px need far fewer
    parameters than a language model to saturate a 10-30k pair dataset
    (see SPEC.md #2, the "ograniczona skala" trade-off).
    """

    def __init__(self, base_channels=64, channel_mults=(1, 2, 4), text_dim=512, cond_dim=256):
        super().__init__()
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.text_proj = nn.Linear(text_dim, cond_dim)

        self.in_conv = nn.Conv2d(6, base_channels, 3, padding=1)

        chs = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []
        in_ch = base_channels
        for i, ch in enumerate(chs):
            self.down_blocks.append(nn.ModuleList([
                FiLMResBlock(in_ch, ch, cond_dim),
                FiLMResBlock(ch, ch, cond_dim),
            ]))
            self.skip_channels.append(ch)
            in_ch = ch
            self.downsamples.append(Downsample(ch) if i < len(chs) - 1 else nn.Identity())

        self.mid_block1 = FiLMResBlock(in_ch, in_ch, cond_dim)
        self.mid_attn = SelfAttention2d(in_ch)
        self.mid_block2 = FiLMResBlock(in_ch, in_ch, cond_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            skip_ch = self.skip_channels[i]
            self.up_blocks.append(nn.ModuleList([
                FiLMResBlock(in_ch + skip_ch, ch, cond_dim),
                FiLMResBlock(ch + skip_ch, ch, cond_dim),
            ]))
            in_ch = ch
            self.upsamples.append(Upsample(ch) if i > 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_conv = nn.Conv2d(in_ch, 3, 3, padding=1)

    def forward(self, x, t, text_emb):
        """x: [B,6,H,W] (noisy_after || before). t: [B] long. text_emb: [B,text_dim]."""
        cond = self.time_mlp(timestep_embedding(t, self.base_channels)) + self.text_proj(text_emb)

        h = self.in_conv(x)
        skips = []
        for (b1, b2), down in zip(self.down_blocks, self.downsamples):
            h = b1(h, cond)
            h = b2(h, cond)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        for (b1, b2), up in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            h = b1(torch.cat([h, skip], dim=1), cond)
            h = b2(torch.cat([h, skip], dim=1), cond)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))
