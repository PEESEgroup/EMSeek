import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class GatedAffineMoE(nn.Module):
    def __init__(self, n_experts: int, hidden: int = 32, residual_scale: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.attn_q = nn.Linear(n_experts, hidden, bias=False)
        self.attn_k = nn.Parameter(torch.randn(n_experts, hidden))
        self.gate_mlp = nn.Sequential(
            nn.Linear(n_experts, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts)
        )
        self.scale = nn.Parameter(torch.ones(n_experts))
        self.bias = nn.Parameter(torch.zeros(n_experts))
        self.residual_scale = residual_scale
    def forward(self, _X_unused: torch.Tensor, base_preds: torch.Tensor):
        q = self.attn_q(base_preds)
        attn_logits = torch.einsum('bh,eh->be', q, self.attn_k) / math.sqrt(q.size(-1))
        gate_input = F.relu(attn_logits)
        gate_logits = self.gate_mlp(gate_input)
        w = F.softmax(gate_logits, dim=-1)
        y_cal = self.scale * base_preds + self.bias
        out = torch.sum(w * y_cal, dim=-1, keepdim=True)
        residual = self.residual_scale * torch.mean(base_preds, dim=-1, keepdim=True)
        y_hat = out + residual
        return y_hat, w, y_cal