"""
Praman Brain Ctx — LLaMA-style SLM with Recurrent Memory (RMT-style).

The conversation is processed in fixed segments of `seg_len` tokens. Each
segment sees:

    [ read_mem (n_mem) | segment tokens (<= seg_len) | write_mem (n_mem) ]

with a causal mask. Tokens attend to the read-memory slots (state carried
from all previous segments) plus earlier tokens in the segment. The write slots
sit at the end, see everything, and their final hidden states become the
read-memory for the next segment. Memory is a fixed-size vector state, so the
model can follow conversations of unbounded length at constant cost per token.

Kept separate from ../model.py so the original single-turn model stays intact.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class CtxConfig:
    vocab_size: int = 1009
    seg_len: int = 512
    n_mem: int = 16
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bptt_segments: int = 4      # gradients flow back through this many segments


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        return (self.weight * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


def precompute_rope(head_dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(end).float()
    ang = torch.outer(t, freqs)
    return torch.cos(ang), torch.sin(ang)       # (end, head_dim/2) each


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); rotate pairs (even, odd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 32 * ((hidden + 31) // 32)
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.n1(x), cos, sin))
        return x + self.drop(self.mlp(self.n2(x)))


class CtxGPT(nn.Module):
    def __init__(self, cfg: CtxConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.mem_init = nn.Parameter(torch.randn(cfg.n_mem, cfg.n_embd) * 0.02)
        self.mem_write = nn.Parameter(torch.randn(cfg.n_mem, cfg.n_embd) * 0.02)
        self.mem_norm = RMSNorm(cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying
        cos, sin = precompute_rope(cfg.n_embd // cfg.n_head, cfg.seg_len + 2 * cfg.n_mem)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():            # scaled init for residual projections
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def init_memory(self, batch_size):
        return self.mem_init.unsqueeze(0).expand(batch_size, -1, -1)

    def segment(self, idx, mem):
        """idx: (B, T<=seg_len), mem: (B, M, C) -> logits (B, T, V), new_mem (B, M, C)."""
        B, T = idx.shape
        M = self.cfg.n_mem
        x = torch.cat([mem, self.tok_emb(idx), self.mem_write.unsqueeze(0).expand(B, -1, -1)], dim=1)
        L = x.size(1)
        cos, sin = self.cos[:L], self.sin[:L]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        new_mem = self.mem_norm(x[:, M + T:])
        logits = self.head(self.norm(x[:, M:M + T]))
        return logits, new_mem

    def forward(self, idx, targets=None):
        """Full-sequence training pass. idx/targets: (B, N) with N any length; targets -1 = ignore."""
        B, N = idx.shape
        S = self.cfg.seg_len
        mem = self.init_memory(B)
        total_loss, total_n = idx.new_zeros((), dtype=torch.float), 0
        all_logits = []
        for si, start in enumerate(range(0, N, S)):
            if si > 0 and si % self.cfg.bptt_segments == 0:
                mem = mem.detach()                      # truncated BPTT across segments
            logits, mem = self.segment(idx[:, start:start + S], mem)
            if targets is None:
                all_logits.append(logits[:, -1:])
                continue
            tgt = targets[:, start:start + S]
            n = int((tgt != -1).sum())
            if n:
                total_loss = total_loss + F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-1, reduction="sum")
                total_n += n
        if targets is None:
            return all_logits[-1], None
        return None, total_loss / max(total_n, 1)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=80, stop_ids=()):
        """Greedy decoding with recurrent memory. Segment boundaries match training (multiples of seg_len)."""
        dev = self.cos.device
        S = self.cfg.seg_len
        mem = self.init_memory(1)
        n_full = len(ids) // S
        if n_full and len(ids) % S == 0:
            n_full -= 1                                  # keep a non-empty current segment
        for i in range(n_full):
            _, mem = self.segment(torch.tensor([ids[i * S:(i + 1) * S]], device=dev), mem)
        cur = list(ids[n_full * S:])
        out = []
        for _ in range(max_new_tokens):
            logits, new_mem = self.segment(torch.tensor([cur], device=dev), mem)
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            if nxt in stop_ids:
                break
            if len(cur) == S:
                mem, cur = new_mem, [nxt]
            else:
                cur.append(nxt)
        return out
