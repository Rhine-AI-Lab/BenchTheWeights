"""Minimal DiT (Peebles & Xie, 2023) matching the `simillusion/dit-s4-cifar10` checkpoint.

Inference-only port of `simulusion.models.diffusion.DiT` (MIT License,
https://github.com/ParamThakkar123/simulusion @ a26833875449a4de68a23e9f4560e33d40dc0d4c),
with the training code and the einops dependency removed. Parameter names are
kept identical so the published state dict loads strictly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    timesteps = timesteps.reshape(-1)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return F.pad(emb, (0, 1)) if dim % 2 else emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    coords = torch.arange(grid_size, dtype=torch.float32)
    gh, gw = torch.meshgrid(coords, coords, indexing="ij")
    return torch.cat(
        [
            sinusoidal_time_embedding(gh.reshape(-1), embed_dim // 2),
            sinusoidal_time_embedding(gw.reshape(-1), embed_dim // 2),
        ],
        dim=1,
    )


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.fc1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, t):
        return self.fc2(self.act(self.fc1(sinusoidal_time_embedding(t, self.frequency_embedding_size))))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)  # +1: CFG null token

    def forward(self, labels):
        return self.embedding_table(labels)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer("pos", get_2d_sincos_pos_embed(embed_dim, self.grid_size).unsqueeze(0))

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2) + self.pos


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d // n_heads
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (w(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2) for w in (self.W_q, self.W_k, self.W_v))
        ctx = F.scaled_dot_product_attention(q, k, v)
        return self.W_o(ctx.transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(mlp_ratio * d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(approximate="tanh"), nn.Dropout(0.0), nn.Linear(hidden, d_model), nn.Dropout(0.0)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), s1, sc1))
        return x + g2.unsqueeze(1) * self.ff(modulate(self.norm2(x), s2, sc2))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=384, depth=12, heads=6, num_classes=10, learn_sigma=True):
        super().__init__()
        self.in_channels, self.patch_size = in_channels, patch_size
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.t_embedder = TimestepEmbedder(d_model)
        self.y_embedder = LabelEmbedder(num_classes, d_model)
        self.patchify = PatchEmbed(img_size, patch_size, in_channels, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, heads) for _ in range(depth)])
        self.final_layer = FinalLayer(d_model, patch_size, self.out_channels)

    def forward(self, x, t, y):
        c = self.t_embedder(t) + self.y_embedder(y)
        x = self.patchify(x)
        for block in self.transformer_blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        g, p, C = self.patchify.grid_size, self.patch_size, self.out_channels
        # "b (h w) (p q c) -> b c (h p) (w q)"
        x = x.view(x.shape[0], g, g, p, p, C).permute(0, 5, 1, 3, 2, 4)
        return x.reshape(x.shape[0], C, g * p, g * p)
