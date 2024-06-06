import torch.nn as nn
import torch
import copy
import math
import time
from .Attention import MultiHeadAttention


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class SensorAttention(nn.Module):
    def __init__(self, sensor_dim, sensor_heads):
        super(SensorAttention, self).__init__()

        self.multi_head_attention = MultiHeadAttention(sensor_dim, sensor_heads)

    def forward(self, x):
        # x: batch_size x time_window x sensor_dim
        attention = self.multi_head_attention(x, x, x)

        return attention


class TimeAttention(nn.Module):
    def __init__(self, time_dim, time_heads):
        super(TimeAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(time_dim, time_heads)

    def forward(self, x):
        attention = self.multi_head_attention(x, x, x)
        return attention


# 构建子层连接结构
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, sublayer):
        return x + self.alpha * self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, sensor_dim, time_dim, sensor_attn, time_attn, dropout):
        super(EncoderLayer, self).__init__()
        self.sensor_attention = sensor_attn
        self.time_attention = time_attn
        self.sublayer1 = SublayerConnection(sensor_dim, dropout)
        self.sublayer2 = SublayerConnection(time_dim, dropout)


    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.sublayer2(x, self.time_attention)
        x = x.transpose(1, 2)
        x = self.sublayer1(x, self.sensor_attention)

        return x


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class InputBlocks(nn.Module):
    def __init__(self, sensor_dim, time_dim, output_dim=48, sensor_heads=11, time_heads=2, num_input_layer=1, dropout=0.2):
        super(InputBlocks, self).__init__()
        self.positional_encoding = PositionalEncoding(sensor_dim, dropout)
        sensor_attn = SensorAttention(sensor_dim, sensor_heads)
        time_attn = TimeAttention(time_dim, time_heads)
        encoder_layer = EncoderLayer(sensor_dim, time_dim, sensor_attn, time_attn, dropout)
        self.encoder = Encoder(encoder_layer, num_input_layer)
        self.dropout = nn.Dropout(p=dropout)
        self.fc_out = nn.Linear(sensor_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):

        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.fc_out(x)
        x = self.dropout(x)
        x = self.relu(x)

        return x