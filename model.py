import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50304 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True 

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "The number of heads must evenly divide the embedding dimension."
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        print("BBBBBBBBBBBB")

        # create a persistent lower triangular matrix for the mask included in the state_dict 
        # but not a tuned parameter
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        print("AAAAAAA")
        B, L, C = x.size()
        
        # retrieve the Q, K, V layers from self.c_attn(x) 
        s = self.c_attn(x)
        Q, K, V = s[:, :, :C], s[:, :, C:2*C], s[:, :, 2*C:]
        # implement getting the multihead attention layers
        # K = ...
        # Q = ...
        # V = ...
        H = self.n_head
        K = K.view(B, L, H, C // H).permute(0, 2, 1, 3)  # (B, H, L, C // H)
        Q = Q.view(B, L, H, C // H).permute(0, 2, 1, 3)  # (B, H, L, C // H)
        V = V.view(B, L, H, C // H).permute(0, 2, 1, 3)  # (B, H, L, C // H)
        # These conditions should be true. 
        print(K.shape)
        # assert(K.shape == torch.tensor.Size([B, self.n_head, L, C // self.n_head]) )
        # manual implementation of attention 
        # Interact queries and keys, scale it by the embedding dimension  
        # att = ...
        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(C // H))  # Shape: (B, H, L, L)
        # Mask it 
        # att = ... 
        att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        # Apply softmax 
        # att = ...
        att = F.softmax(att, dim=-1)
        # We have implemented a dropout layer for you. Uncomment it after you are finished. 
        att = self.attn_dropout(att) 
        # have it interact with the value keys 
        # y = ...
        y = att @ V
        assert(y.shape == torch.tensor.size([B, self.n_heads, L, C // self.n_heads]))
        # re-assemble all head outputs side by side. Uncomment when you are finished. 
        y = y.transpose(1, 2).contiguous().view(B, L, C) 
        # output projection. Uncomment when you are finished. 
        y = self.resid_dropout(self.c_proj(y))
        # modify the return statement to output y
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.lin_layer1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.lin_layer2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.lin_layer1(x)
        x = self.gelu(x)
        x = self.lin_layer2(x)
        x = self.dropout(x)
        return x

class AttentionBlock(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([AttentionBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos) 
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
