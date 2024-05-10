import os
import numpy as np
import pandas as pd
import random
import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', num_classes=21):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # 将targets转换为one-hot编码形式
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')
        pt = torch.exp(-BCE_loss)  # 防止当概率为0时计算log(0)导致的nan
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        

class loadOpp(Dataset):
    def __init__(self, features_folder, labels_folder, window_size=10, step_size=1):
        """
        features_folder: 存放时间序列特征文件的文件夹路径
        labels_folder: 存放时间序列标签文件的文件夹路径
        window_size: 滑动窗口的大小（以时间步为单位）
        step_size: 滑动窗口的步长
        """
        self.files_data = []  # 存储每个文件的特征和标签
        self.window_size = window_size
        self.step_size = step_size
        self.index_map = []  # 存储每个样本对应的文件索引和窗口起始位置

        # 遍历features文件夹，寻找匹配的labels文件
        for features_file in os.listdir(features_folder):
            if features_file.endswith('.csv'):
                features_file_prefix = features_file[:7]
                labels_file = f"{features_file_prefix}_activity_249.csv"
                labels_file_path = os.path.join(labels_folder, labels_file)

                # 确认labels文件存在
                if os.path.exists(labels_file_path):
                    # 读取数据
                    current_features = pd.read_csv(os.path.join(features_folder, features_file)).values
                    current_labels = pd.read_csv(labels_file_path).values

                    # 计算当前文件的滑动窗口数量并更新index_map
                    num_windows = (len(current_features) - window_size) // step_size + 1
                    for window_start in range(0, num_windows * step_size, step_size):
                        self.index_map.append((len(self.files_data), window_start))

                    # 添加文件数据
                    self.files_data.append((current_features, current_labels))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 根据idx获取对应的文件索引和窗口起始位置
        file_idx, window_start = self.index_map[idx]
        features, labels = self.files_data[file_idx]  # 获取对应的特征和标签数据

        # 计算窗口结束位置，并切片获取数据
        window_end = window_start + self.window_size
        features = features[window_start:window_end, 1:]  # 提取特征
        label = labels[window_end - 1, 1]  # 取最后一个标签为标签

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor


    
class loadOpp17(Dataset):
    def __init__(self, features_folder, labels_folder, window_size=10, step_size=1):
        """
        features_folder: 存放时间序列特征文件的文件夹路径
        labels_folder: 存放时间序列标签文件的文件夹路径
        window_size: 滑动窗口的大小（以时间步为单位）
        step_size: 滑动窗口的步长
        """
        self.files_data = []  # 存储每个文件的特征和标签
        self.window_size = window_size
        self.step_size = step_size
        self.index_map = []  # 存储每个样本对应的文件索引和窗口起始位置

        # 遍历features文件夹，寻找匹配的labels文件
        for features_file in os.listdir(features_folder):
            if features_file.endswith('.csv'):
                features_file_prefix = features_file[:7]
                labels_file = f"{features_file_prefix}_activity_249.csv"
                labels_file_path = os.path.join(labels_folder, labels_file)

                # 确认labels文件存在
                if os.path.exists(labels_file_path):
                    # 读取数据
                    current_features = pd.read_csv(os.path.join(features_folder, features_file)).values
                    current_labels = pd.read_csv(labels_file_path).values

                    # 计算当前文件的滑动窗口数量并更新index_map
                    num_windows = (len(current_features) - window_size) // step_size + 1
                    for window_start in range(0, num_windows * step_size, step_size):
                        window_end = window_start + window_size
                        if current_labels[window_end - 1, 1] != 0:  # 如果最后一个标签不是0
                            self.index_map.append((len(self.files_data), window_start))

                    # 添加文件数据
                    self.files_data.append((current_features, current_labels))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 根据idx获取对应的文件索引和窗口起始位置
        file_idx, window_start = self.index_map[idx]
        features, labels = self.files_data[file_idx]  # 获取对应的特征和标签数据

        # 计算窗口结束位置，并切片获取数据
        window_end = window_start + self.window_size
        features = features[window_start:window_end, 1:]  # 提取特征
        label = labels[window_end - 1, 1] - 1  # 取最后一个标签为标签，并减一

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor


class loadUCI(Dataset):
    def __init__(self, features_folder, labels_folder, window_size=10, step_size=1):
        """
        features_folder: 存放时间序列特征文件的文件夹路径
        labels_folder: 存放时间序列标签文件的文件夹路径
        window_size: 滑动窗口的大小（以时间步为单位）
        step_size: 滑动窗口的步长
        """
        self.files_data = []  # 存储每个文件的特征和标签
        self.window_size = window_size
        self.step_size = step_size
        self.index_map = []  # 存储每个样本对应的文件索引和窗口起始位置

        # 遍历features文件夹，寻找匹配的labels文件
        for features_file in os.listdir(features_folder):
            if features_file.endswith('.csv'):
                features_file_prefix = features_file[:2]
                labels_file = f"{features_file_prefix}_activity.csv"
                labels_file_path = os.path.join(labels_folder, labels_file)

                # 确认labels文件存在
                if os.path.exists(labels_file_path):
                    # 读取数据
                    current_features = pd.read_csv(os.path.join(features_folder, features_file)).values
                    current_labels = pd.read_csv(labels_file_path).values

                    # 计算当前文件的滑动窗口数量并更新index_map
                    num_windows = (len(current_features) - window_size) // step_size + 1
                    for window_start in range(0, num_windows * step_size, step_size):
                        self.index_map.append((len(self.files_data), window_start))

                    # 添加文件数据
                    self.files_data.append((current_features, current_labels))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 根据idx获取对应的文件索引和窗口起始位置
        file_idx, window_start = self.index_map[idx]
        features, labels = self.files_data[file_idx]  # 获取对应的特征和标签数据

        # 计算窗口结束位置，并切片获取数据
        window_end = window_start + self.window_size
        features = features[window_start:window_end, 1:]  # 提取特征
        label = labels[window_end - 1, 1]  # 取最后一个标签为标签

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor

class loadHAD(Dataset):
    def __init__(self, features_folder, labels_folder, window_size=10, step_size=1):
        """
        features_folder: 存放时间序列特征文件的文件夹路径
        labels_folder: 存放时间序列标签文件的文件夹路径
        window_size: 滑动窗口的大小（以时间步为单位）
        step_size: 滑动窗口的步长
        """
        self.files_data = []  # 存储每个文件的特征和标签
        self.window_size = window_size
        self.step_size = step_size
        self.index_map = []  # 存储每个样本对应的文件索引和窗口起始位置

        # 遍历features文件夹，寻找匹配的labels文件
        for features_file in os.listdir(features_folder):
            if features_file.endswith('.csv'):
                features_file_prefix = features_file[:2]
                labels_file = f"{features_file_prefix}_activity.csv"
                labels_file_path = os.path.join(labels_folder, labels_file)

                # 确认labels文件存在
                if os.path.exists(labels_file_path):
                    # 读取数据
                    current_features = pd.read_csv(os.path.join(features_folder, features_file)).values
                    current_labels = pd.read_csv(labels_file_path).values

                    # 计算当前文件的滑动窗口数量并更新index_map
                    num_windows = (len(current_features) - window_size) // step_size + 1
                    for window_start in range(0, num_windows * step_size, step_size):
                        self.index_map.append((len(self.files_data), window_start))

                    # 添加文件数据
                    self.files_data.append((current_features, current_labels))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 根据idx获取对应的文件索引和窗口起始位置
        file_idx, window_start = self.index_map[idx]
        features, labels = self.files_data[file_idx]  # 获取对应的特征和标签数据

        # 计算窗口结束位置，并切片获取数据
        window_end = window_start + self.window_size
        features = features[window_start:window_end, 0:]  # 提取特征
        label = labels[window_end - 1, 0]  # 取最后一个标签为标签

        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor

