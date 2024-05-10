import torch
import torch.nn as nn
import torch.nn.functional as F

class SensorPre(nn.Module):
    def __init__(self, dim, n_filters, kernel_size, dilation_rate):
        super(SensorPre, self).__init__()
        self.conv_1 = nn.Conv2d(1, n_filters, kernel_size=kernel_size,
                                dilation=dilation_rate, padding='same', bias=False)
        self.relu = nn.ReLU()
        self.conv_f = nn.Conv2d(n_filters, 1, kernel_size=1, padding='same', bias=False)
        self.ln = nn.LayerNorm(dim)  # Adjust the normalized_shape according to your input size

    def forward(self, x):
        # PyTorch expects batch, channel, height, width (BCHW)
        x = self.ln(x)
        x1 = x.unsqueeze(1)  # Increase the dimension for channels if not already present
        x1 = self.relu(self.conv_1(x1))
        x1 = self.conv_f(x1)
        x1 = F.softmax(x1, dim=3)  # Apply softmax along the correct dimension
        x1 = x1.view(x.shape)  # Reshape x1 to match the dimensions of x
        return x * x1, x1