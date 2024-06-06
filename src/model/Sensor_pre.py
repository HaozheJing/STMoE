import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorPre(nn.Module):
    """传感器预处理单元"""
    def __init__(self, dim, n_filters, kernel_size, dilation_rate):
        super(SensorPre, self).__init__()
        self.conv_1 = nn.Conv2d(1, n_filters, kernel_size=kernel_size,
                                dilation=dilation_rate, padding='same', bias=False)
        self.relu = nn.ReLU()
        self.conv_f = nn.Conv2d(n_filters, 1, kernel_size=1, padding='same', bias=False)
        self.ln = nn.LayerNorm(dim) 

    def forward(self, x):
        x = self.ln(x)
        x1 = x.unsqueeze(1) 
        x1 = self.relu(self.conv_1(x1))
        x1 = self.conv_f(x1)
        x1 = F.softmax(x1, dim=3) 
        x1 = x1.view(x.shape)  
        return x * x1, x1