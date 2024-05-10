import torch.nn as nn
import torch
import copy
import math
from .Attention import MultiHeadAttention


def clones(module, N):
    # module: 代表要克隆的目标网络层
    # N: 将module克隆几个
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ConditionAttention(nn.Module):
    def __init__(self, embed_size, heads, position_dim, position_indexes):
        # positon_dim: 打算嵌入的维度

        super(ConditionAttention, self).__init__()
        self.embed_size = embed_size
        self.position_indexes = position_indexes
        self.heads = heads
        self.head_dim = embed_size // heads
        self.position_dim = position_dim

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        assert (
                self.position_dim % self.heads == 0
        ), "Position dimension must be divisible by the number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim + int(self.position_dim / self.heads), self.head_dim, bias=False)

        # 用于位置信息的线性层，输入维度为20（独热编码的大小），输出维度为position_dim
        self.position_embedding = nn.Linear(self.position_indexes, self.position_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, position):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 通过线性层转换独热编码的位置信息，这里position假设已经是N x query_len x 20
        position_embeddings = self.position_embedding(position)  # 结果形状将是 N x query_len x position_dim

        queries_with_position = torch.cat((query, position_embeddings), dim=2)  # 正确的拼接
        # 接下来，根据头的数量分割queries_with_position，然后对values和keys做相应操作
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries_with_position = queries_with_position.reshape(N, query_len, self.heads,
                                                              self.head_dim + int(self.position_dim / self.heads))

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries_with_position)

        # 后续步骤与之前相同
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class SensorAttention(nn.Module):
    def __init__(self, sensor_dim, sensor_heads, position_dim, position_indexes):
        super(SensorAttention, self).__init__()
        # 注意新增了position_dim参数
        self.multi_head_attention = ConditionAttention(sensor_dim, sensor_heads, position_dim, position_indexes)

    def forward(self, x, position_indexes):
        attention = self.multi_head_attention(x, x, x, position_indexes)
        return attention


class TimeAttention(nn.Module):
    def __init__(self, time_dim, time_heads):
        super(TimeAttention, self).__init__()
        # 注意新增了position_dim参数
        self.multi_head_attention = MultiHeadAttention(time_dim, time_heads)

    def forward(self, x):
        attention = self.multi_head_attention(x, x, x)
        return attention


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

        # 参数化残差连接
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, sublayer):
        # 将LayerNorm移到子层前面，并引入参数化残差连接
        return x + self.alpha * self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, sensor_dim, time_dim, sensor_attn, time_attn, dropout):
        super(EncoderLayer, self).__init__()
        self.sensor_attention = sensor_attn
        self.time_attention = time_attn
        self.sublayer1 = SublayerConnection(sensor_dim, dropout)
        self.sublayer2 = SublayerConnection(time_dim, dropout)

    def forward(self, x, position_indexes):
        x = self.sublayer1(x, lambda _x: self.sensor_attention(_x, position_indexes))
        x = x.transpose(1, 2)  # 维度转换
        x = self.sublayer2(x, lambda _x: self.time_attention(_x))
        x = x.transpose(1, 2)  # 维度转换回来
        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x, position_indexes):
        for layer in self.layers:
            x = layer(x, position_indexes)
        return x


class InputBlocks2(nn.Module):
    def __init__(self, sensor_dim, time_dim, position_dim, position_indexes=9, sensor_heads=6, time_heads=2,
                 num_input_layer=1, dropout=0.1):
        super(InputBlocks2, self).__init__()
        self.positional_encoding = PositionalEncoding(sensor_dim, dropout)
        sensor_attn = SensorAttention(sensor_dim, sensor_heads, position_dim, position_indexes)
        time_attn = TimeAttention(time_dim, time_heads)
        encoder_layer = EncoderLayer(sensor_dim, time_dim, sensor_attn, time_attn, dropout)
        self.position_indexes = position_indexes
        self.encoder = Encoder(encoder_layer, num_input_layer)

    def forward(self, x):
        position = x[:, :, -self.position_indexes:]
        x = x[:, :, :-self.position_indexes]
        x = self.positional_encoding(x)
        return self.encoder(x, position)


