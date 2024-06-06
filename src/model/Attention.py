import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """多头自注意力机制的实现"""
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize Linear layers using Xavier initialization
        nn.init.xavier_uniform_(self.values.weight)
        nn.init.xavier_uniform_(self.keys.weight)
        nn.init.xavier_uniform_(self.queries.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Calculate attention using einsum
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out




class AttentionWithContext(nn.Module):
    """
    没有用这个，这是参考以前学者的一个实现。
    """
    def __init__(self, input_dim, bias=True, return_attention=False):
        super(AttentionWithContext, self).__init__()
        self.input_dim = input_dim
        self.return_attention = return_attention
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.u = nn.Parameter(torch.Tensor(input_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(input_dim))

        nn.init.xavier_uniform_(self.W)
        nn.init.uniform_(self.u, -0.1, 0.1)  
        if bias:
            nn.init.zeros_(self.b)

    def forward(self, x):
        
        uit = torch.tensordot(x, self.W, dims=([-1], [0]))
        if self.bias:
            uit += self.b
        uit = torch.tanh(uit)

        ait = torch.tensordot(uit, self.u, dims=([-1], [0]))
        a = torch.exp(ait)
        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-10)

        weighted_input = x * a.unsqueeze(-1)
        result = torch.sum(weighted_input, dim=1)

        if self.return_attention:
            return result, a
        return result
