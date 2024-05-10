import torch
import matplotlib.pyplot as plt
import seaborn as sns
from DataLoader import loadOpp17
from torch.utils.data import DataLoader
import os

# 定义数据文件夹路径
features_folder = r'data/OPP/train/input'
labels_folder = r'data/OPP/train/label'

print("初始化数据集...")
window_size = 16
step_size = 1
train_dataset = loadOpp17(features_folder, labels_folder, window_size, step_size)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print("数据集初始化完成。")

# 加载已训练的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("加载模型...")
model = torch.load('save/model/OPP_model.pth', map_location=device)
model.eval()  # 设置为评估模式
print("模型加载并设置为评估模式。")

# 初始化专家选择计数矩阵
num_experts = 4  # 模型中专家的数量
num_classes = 17  # 假设数据集类别数量可直接获取
expert_count_matrix = torch.zeros(num_classes, num_experts, dtype=torch.int)

# 从训练集的 DataLoader 中提取几个 batch 的数据并进行处理
num_batches = 50  # 您可以根据需要调整提取的 batch 数量
print(f"正在从 DataLoader 提取前 {num_batches} 个 batch 的数据进行分析...")

for data, target in train_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    gating_data = model.get_gating_data()  # 获取门控数据

    # 更新专家选择计数矩阵
    top_experts_1 = gating_data['top_experts']['index_1'].cpu()  # 使用第一顶级专家的索引，并确保它在 CPU 上
    top_experts_2 = gating_data['top_experts']['index_2'].cpu()  # 第二顶级专家的索引
    target = target.cpu()  # 确保目标也在 CPU 上
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
plt.savefig('save/OPP/Experts_Activities.png')
