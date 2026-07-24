"""Gaussian diffusion schedule: linear beta_t noise schedule (DDPM) with a
DDIM sampler for fast few-step inference. Hand-rolled rather than imported —
same reasoning as MicroG: understanding every layer matters as much as a
working model (see SPEC.md).
"""

import torch


class DiffusionSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    def q_sample(self, x0, t, noise=None):
        """Forward process: x_t = sqrt(acp_t) * x0 + sqrt(1 - acp_t) * noise."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1macp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_acp * x0 + sqrt_1macp * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model, before, text_emb, steps=100, device="cpu"):
        """Deterministic (eta=0) DDIM sampler, `steps` << self.timesteps.

        `steps` trades quality for the seconds-per-edit budget on a CPU/MPS
        Mac (SPEC.md #5 leaves the exact count open until real inference is
        timed). Measured 2026-07-22 on the cross-attention checkpoint: 20
        steps gave a washed-out result that looked nearly identical
        regardless of prompt; 100 steps on the SAME checkpoint revealed real
        structure and color the 20-step version was hiding. Don't judge
        output quality below ~50.
        """
        b = before.shape[0]
        seq = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long, device=device)
        x = torch.randn_like(before)
        acp = self.alphas_cumprod.to(device)
        for i, t in enumerate(seq):
            t_batch = t.repeat(b)
            pred_noise = model(torch.cat([x, before], dim=1), t_batch, text_emb)
            a_t = acp[t]
            x0_pred = ((x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()).clamp(-1, 1)
            if i == len(seq) - 1:
                x = x0_pred
                break
            a_prev = acp[seq[i + 1]]
            x = a_prev.sqrt() * x0_pred + (1 - a_prev).sqrt() * pred_noise
        return x
