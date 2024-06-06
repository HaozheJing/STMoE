import torch
import matplotlib.pyplot as plt
import seaborn as sns
from DataLoader import loadUCI
from torch.utils.data import DataLoader
import os


features_folder = r'data/UCI/train/input'
labels_folder = r'data/UCI/train/label'

print("初始化数据集...")
window_size = 128
step_size = 1
train_dataset = loadUCI(features_folder, labels_folder, window_size, step_size)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
print("数据集初始化完成。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("加载模型...")
model = torch.load('save/model/UCI_model.pth', map_location=device)
model.eval()  
print("模型加载并设置为评估模式。")

num_experts = 4  
num_classes = 6  
expert_count_matrix = torch.zeros(num_classes, num_experts, dtype=torch.int)

# 从训练集的 DataLoader 中提取几个 batch 的数据并进行处理
num_batches = 50 
print(f"正在从 DataLoader 提取前 {num_batches} 个 batch 的数据进行分析...")

for data, target in train_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    gating_data = model.get_gating_data() 

    top_experts_1 = gating_data['top_experts']['index_1'].cpu()
    top_experts_2 = gating_data['top_experts']['index_2'].cpu()  
    target = target.cpu()  
    for i in range(data.size(0)):
        expert_count_matrix[target[i], top_experts_1[i]] += 1
        expert_count_matrix[target[i], top_experts_2[i]] += 1

    num_batches -= 1
    if num_batches == 0:
        break

print("数据处理完成，正在生成热力图...")

# 绘制热力图
plt.figure(figsize=(12, 8))
sns.heatmap(expert_count_matrix.numpy(), annot=True, cmap="YlGnBu", fmt="d",
            xticklabels=["Expert" + str(i) for i in range(num_experts)],
            yticklabels=["Activity" + str(i) for i in range(num_classes)])
plt.xlabel('Experts')
plt.ylabel('Activities')
plt.savefig('save/UCI/Experts_Activities.png')
