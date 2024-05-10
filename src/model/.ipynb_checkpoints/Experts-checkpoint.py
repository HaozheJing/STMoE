import torch
from torch import nn
import math
from inspect import isfunction


def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)


def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)


def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val


class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_


class Experts(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=16,
                 hidden_dim=None,
                 activation=GELU,
                 dropout=0.2):
        super().__init__()

        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.dropout = nn.Dropout(p=dropout)
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.act = activation()

    def forward(self, x):
        hidden = torch.einsum('...nd,...dh->...nh', x, self.w1)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        out = torch.einsum('...nh,...hd->...nd', hidden, self.w2)
        out = self.dropout(out)
        return out
