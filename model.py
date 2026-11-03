"""
Praman Brain Mini — Model Architecture (LLaMA-style)
Upgraded to use RMSNorm, RoPE, and SwiGLU.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 100284   # tiktoken cl100k_base + our 7 special tokens
    block_size: int = 128      # context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 32           # keeps total params ~3M (mostly in embedding table)
    dropout: float = 0.1
    bias: bool = False         # LLaMA standard is no bias
    share_layers: bool = True  # Loop Engineering

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def apply_rotary_emb(x, freqs_cis):
    # x: (B, T, n_head, head_dim)
    # freqs_cis: (T, head_dim/2, 2) -> complex
    # Simplified implementation for PyTorch 2.0+
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (1, T, 1, head_dim/2)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wk = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wv = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim)
        
        q = apply_rotary_emb(q, freqs_cis[:T])
        k = apply_rotary_emb(k, freqs_cis[:T])
        
        q = q.transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y))

class SwiGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.feed_forward = SwiGLU_MLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        if getattr(config, 'share_layers', False):
            self.shared_block = Block(config)
            self.layers = None
        else:
            self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
            self.shared_block = None
        self.norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.tok_embeddings.weight = self.output.weight
        
        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, config.block_size)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, "sequence longer than block_size"
        
        x = self.drop(self.tok_embeddings(idx))
        if self.layers is None:
            for _ in range(self.config.n_layer):
                x = self.shared_block(x, self.freqs_cis)
        else:
            for layer in self.layers:
                x = layer(x, self.freqs_cis)
        x = self.norm(x)
        
        if targets is not None:
            logits = self.output(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        logits = self.output(x[:, [-1], :])
        return logits, None
