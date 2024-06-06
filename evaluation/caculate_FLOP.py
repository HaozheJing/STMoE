import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time
from torchinfo import summary
from torchprofile import profile_macs
from DataLoader import loadOpp, loadUCI, loadHAD, loadOpp17

# 加载模型和数据
model = torch.load('save/model/UCI_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

test_features_folder = r'data/UCI/test/input'
test_labels_folder = r'data/UCI/test/label'
dataset = loadUCI(test_features_folder, test_labels_folder, window_size=128, step_size=1)
batch_size = 128
print("Loading data...")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 选择用于计算 FLOPs 的输入
example_inputs, _ = next(iter(dataloader))
example_inputs = example_inputs.to(device)

# 计算模型的 MACs
model.eval()
with torch.no_grad():
    macs = profile_macs(model, example_inputs)
    flops = 2 * macs 
print(f"Model FLOPs: {flops}")

# 计算 FLOPs
print("Calculating model FLOPs...")
summary(model, input_size=(batch_size, 128, 9)) 

# 计算推理速度
print("Measuring inference speed...")
model.eval()
with torch.no_grad():
    total_samples = 0
    total_time = 0
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        start_time = time()
        _ = model(inputs)
        end_time = time()
        total_time += (end_time - start_time)
        total_samples += len(inputs)

average_inference_speed = total_time / total_samples
print(f"Average inference time per sample: {average_inference_speed:.6f} seconds")
