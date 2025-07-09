import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    """GPT-2 model configuration"""

    def __init__(self, block_size, vocab_size=None, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def gpt2(cls, **override_args):
        # Default GPT-2 (small) configuration
        config = {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "embd_pdrop": 0.1,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
        }
        config.update(override_args)
        # vocab_size and block_size are required
        return cls(**config)


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention layer"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network for the Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """A single Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """The full GPT language model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size

        self.transformer = nn.ModuleDict(
            dict(
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        # The following layers are initialized in _init_weights
        self.lm_head = None
        self.transformer.wte = None

    def _init_weights(self):
        """Initialise weights for layers that depend on vocab_size."""
        self.transformer.wte = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        # Weight tying: share the weights between the token embedding and the final linear layer
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Sequence length {t} exceeds block size {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def get_features(self, idx, last_token_only=True):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f"Sequence length {t} exceeds block size {self.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        if last_token_only:
            return x[:, -1, :]  # (B, C)
        else:
            return x  # (B, T, C)

    def get_features_long(self, idx):
        """Get features for a sequence longer than the block size."""
        final_features = []
        # Process the sequence in chunks of block_size
        for i in range(0, idx.size(1), self.block_size):
            chunk = idx[:, i : i + self.block_size]
            features = self.get_features(chunk)
            final_features.append(features)
        return torch.cat(final_features, dim=1)
