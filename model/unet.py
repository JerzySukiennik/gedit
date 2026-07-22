"""Small conditional U-Net — the part of Gedit that IS trained from scratch.
Predicts the noise added to `after`, conditioned on `before` (concatenated on
the channel axis — same img2img formulation InstructPix2Pix itself uses) and
the full per-token CLIP text sequence via cross-attention.

Architecture history: v1 (see git log) used FiLM (global scale/shift from a
single pooled text vector) instead of cross-attention, for simplicity. It
worked for whole-image style/color edits but could not localize object edits
("add a hat" produced a global tone shift, not a hat) — a pooled vector
carries no per-word or spatial information, so the model had no way to know
*where* "hat" should apply. Cross-attention gives each spatial position a
query that can attend to the specific word driving it. Timestep conditioning
stays FiLM (that one's genuinely global — "how much noise" applies to the
whole image equally); only text conditioning moved to cross-attention.
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


class TimeResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, twice, with FiLM (scale, shift) from the
    timestep embedding applied after the first norm — this one genuinely is
    a whole-image quantity ("how much noise to remove"), unlike text. 1x1
    conv on the residual path when the channel count changes.
    """

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
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


class CrossAttention2d(nn.Module):
    """Each spatial position (query) attends over the text token sequence
    (key/value) — this is what lets a specific word ("hat") drive a specific
    region of the image, instead of every word blending into one global
    scale/shift like FiLM did.
    """

    def __init__(self, channels, text_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.norm = nn.GroupNorm(8, channels)
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(text_dim, channels)
        self.to_v = nn.Linear(text_dim, channels)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, text_seq):
        """x: [B,C,H,W]. text_seq: [B,L,text_dim]."""
        b, c, h, w = x.shape
        L = text_seq.shape[1]
        q = self.to_q(self.norm(x)).reshape(b, self.heads, self.head_dim, h * w)
        k = self.to_k(text_seq).reshape(b, L, self.heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.to_v(text_seq).reshape(b, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(self.head_dim), dim=-1)  # b,heads,hw,L
        out = (attn @ v).transpose(-1, -2).reshape(b, c, h, w)  # b,heads,hw,head_dim -> b,c,h,w
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
    level. Cross-attention to the full CLIP text sequence after every
    resolution level's ResBlocks (down and up) plus the bottleneck, so object
    placement can be driven at multiple spatial scales, not just the
    coarsest one. Self-attention (spatial-only) also kept at the bottleneck.
    """

    def __init__(self, base_channels=64, channel_mults=(1, 2, 4), text_dim=512, time_dim=256):
        super().__init__()
        self.base_channels = base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        self.in_conv = nn.Conv2d(6, base_channels, 3, padding=1)

        chs = [base_channels * m for m in channel_mults]
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []
        in_ch = base_channels
        for i, ch in enumerate(chs):
            self.down_blocks.append(nn.ModuleList([
                TimeResBlock(in_ch, ch, time_dim),
                TimeResBlock(ch, ch, time_dim),
            ]))
            self.down_attns.append(CrossAttention2d(ch, text_dim))
            self.skip_channels.append(ch)
            in_ch = ch
            self.downsamples.append(Downsample(ch) if i < len(chs) - 1 else nn.Identity())

        self.mid_block1 = TimeResBlock(in_ch, in_ch, time_dim)
        self.mid_self_attn = SelfAttention2d(in_ch)
        self.mid_cross_attn = CrossAttention2d(in_ch, text_dim)
        self.mid_block2 = TimeResBlock(in_ch, in_ch, time_dim)

        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chs))):
            skip_ch = self.skip_channels[i]
            self.up_blocks.append(nn.ModuleList([
                TimeResBlock(in_ch + skip_ch, ch, time_dim),
                TimeResBlock(ch + skip_ch, ch, time_dim),
            ]))
            self.up_attns.append(CrossAttention2d(ch, text_dim))
            in_ch = ch
            self.upsamples.append(Upsample(ch) if i > 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_conv = nn.Conv2d(in_ch, 3, 3, padding=1)

    def forward(self, x, t, text_seq):
        """x: [B,6,H,W] (noisy_after || before). t: [B] long.
        text_seq: [B,L,text_dim] (full CLIP token sequence, not pooled).
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.base_channels))

        h = self.in_conv(x)
        skips = []
        for (b1, b2), attn, down in zip(self.down_blocks, self.down_attns, self.downsamples):
            h = b1(h, t_emb)
            h = b2(h, t_emb)
            h = attn(h, text_seq)
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_self_attn(h)
        h = self.mid_cross_attn(h, text_seq)
        h = self.mid_block2(h, t_emb)

        for (b1, b2), attn, up in zip(self.up_blocks, self.up_attns, self.upsamples):
            skip = skips.pop()
            h = b1(torch.cat([h, skip], dim=1), t_emb)
            h = b2(torch.cat([h, skip], dim=1), t_emb)
            h = attn(h, text_seq)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))
