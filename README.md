
欢迎测试我的代码！本项目由大数据21景浩哲（2021311654）开发。以下是我的代码的使用方式简单介绍。
@[TOC](目录)

# 环境搭建

本项目的全部代码均基于Python深度学习框架Pytorch实现，各个库版本如下：
|库名称| 版本号 |
|--|--|
|python  |3.10|
|numpy|1.26.3|
| scikit-learn |1.5.0  |
|seaborn|0.13.2|
| torch |2.1.2  |
| tqdm |4.64.1  |


请将终端打开当前项目的根文件夹，使用`pip install -r requirements.txt` 命令进行一键安装

# 快速开始
针对于作业中已经调试好的三个数据集，使用以下命令直接进行训练。请注意，`--dataset`, `--num_classes` 为必须提供的超参数，表示训练哪个数据集，以及该数据集有多少类。

**训练并测试UCI-HAR数据集**
```bash
python train.py --dataset UCI --num_classes 6 --lr 0.005 --window_size 128 --batch_size 128
```

**训练并测试Opportunity数据集**
```bash
python train.py --dataset OPP --num_classes 17 --lr 0.005 --window_size 64 --batch_size 128
```

**训练并测试USC-HAD数据集**
```bash
python train.py --dataset HAD --num_classes 12 --lr 0.01 --window_size 128 --batch_size 128
```
以上，前两个数据集在半小时内能够训练完成，USC-HAD数据集需要用约2小时训练完成。

# 可供选择的其他超参数
请使用 `python train.py --help` 查看全部超参数的定义和用法。超参数采用`dataclass`装饰器进行修饰，全部超参数存储在项目路径`src/hparams.py`下。
# 各文件夹下代码简介
**data文件夹**
该文件中存放已经经过预处理的各个数据集，以及`dataloader.py`文件，用于加载并划分各个数据集的时间窗口

**src文件夹**
该部分为模型各个单元的实现方法，代码命名即为模块的名称

**evaluation文件夹**
该部分为模型评估所用到的代码，包括T-SNE降维可视化、MOE可视化、计算FLOPS等

**save文件夹**
模型、绘图等都会保存在该文件夹内。


