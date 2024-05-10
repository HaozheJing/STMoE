import torch
import os
import random
import numpy as np
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Import custom modules
from src.model.STMoE import STMoE
from data.DataLoader import loadOpp, loadUCI, loadHAD, loadOpp17
from src.hparams import get_args, Config  


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

            
def plot_loss(dataset, loss_history):
    plt.plot(loss_history, label='Training Loss')
    #plt.plot(smooth_loss_history, label='Smooth Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('save/' + dataset + '/loss_plot.png')
    plt.close()            

    
def plot_confusion_matrix(dataset, targets, outputs, class_names):
    # 计算混淆矩阵
    cm = confusion_matrix(targets, outputs)
    
    # 设置图像大小
    plt.figure(figsize=(15, 12))
    
    # 绘制热力图，调整字体大小
    heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='inferno_r',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})  # 调整注释数字的字体大小
    
    # 设置轴标签和标题的字体大小
    plt.xlabel('Predicted labels', fontsize=22)
    plt.ylabel('True labels', fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('save/' + dataset + '/confusion_matrix_heatmap.png')
    plt.close()
    
def save_model(model, save_path, dataset):
    os.makedirs(save_path, exist_ok=True)
    torch.save(model, os.path.join(save_path, dataset + '_model.pth'))

    
def train(model, optimizer, loss_function, train_loader, val_loader, test_loader, device, epochs):
    loss_history = []
    val_loss_history = []
    val_f1_history = []

    for epoch in tqdm(range(epochs), desc="Epochs", colour='green'):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_history.append(loss.item())

        epoch_loss = total_loss / len(train_loader)
        epoch_val_loss, epoch_f1_score = validate(model, val_loader, loss_function, device)
        val_loss_history.append(epoch_val_loss)
        val_f1_history.append(epoch_f1_score)

        print(
            f"Epoch {epoch}: Training Loss {epoch_loss:.4f}, Validation Loss {epoch_val_loss:.4f}, Validation F1 {epoch_f1_score:.4f}")
        targets, outputs = test(model, test_loader, loss_function, device)
        
    return loss_history, val_loss_history, val_f1_history
    

def validate(model, val_loader, loss_function, device):
    model.eval()
    total_val_loss = 0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target.view(-1))
            total_val_loss += loss.item()
            predicted_classes = output.argmax(dim=1)
            all_targets.extend(target.view(-1).cpu().numpy())
            all_outputs.extend(predicted_classes.cpu().numpy())

    epoch_val_loss = total_val_loss / len(val_loader)
    epoch_f1_score = f1_score(all_targets, all_outputs, average='macro')
    return epoch_val_loss, epoch_f1_score

def test(model, data_loader, loss_function, device):
    model.eval()
    test_loss = 0
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_function(output, target.view(-1)).item()
            pred = output.argmax(dim=1, keepdim=True)
            targets_list.extend(target.cpu().view_as(pred).tolist())
            outputs_list.extend(pred.cpu().tolist())

    test_loss /= len(data_loader.dataset)
    print(classification_report(np.concatenate(targets_list), np.concatenate(outputs_list), digits=4))
    epoch_f1_score = f1_score(np.concatenate(targets_list),np.concatenate(outputs_list), average='macro')
    return np.concatenate(targets_list), np.concatenate(outputs_list)


def get_data_loader(args: Config):
    if args.dataset == "OPP":
        load_function = loadOpp17
        ACTIVITIES = ['Open Door 1', 'Open Door 2', 'Close Door 1', 'Close Door 2', 'Open Fridge', 'Close Fridge', 'Open Dishwasher', 'Close Dishwasher', 
                      'Open Drawer 1', 'Close Drawer 1', 'Open Drawer 2', 'Close Drawer 2', 'Open Drawer 3', 'Close Drawer 3', 'Clean Table', 'Drink from Cup', 
                      'Toggle Switch']
        
    elif args.dataset == "HAD":
        load_function = loadHAD
        ACTIVITIES = ['Walking Forward', 'Walking Left', 'Walking Right', 'Walking Upstairs',  'Walking Downstairs',  'Running Forward', 'Jumping Up', 'Sitting', 
                      'Standing',  'Sleeping',  'Elevator Up',  'Elevator Down']
        
    elif args.dataset == "UCI":
        load_function = loadUCI
        ACTIVITIES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING',  'STANDING',  'LAYING']
    
    elif args.dataset == "SPHERE":
        load_function = loadSPHERE
        ACTIVITIES = ['a_ascend', 'a_descend', 'a_jump', 'a_loadwalk', 'a_walk', 'p_bent', 'p_kneel', 'p_lie', 'p_sit', 'p_squat', 'p_stand', 't_bend', 't_kneel_stand', 't_lie_sit', 
                      't_sit_lie', 't_sit_stand', 't_stand_kneel', 'jt_stand_sit', 't_straighten', 't_turn']
        
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(load_function(args.path + args.dataset + '/train/input', args.path + args.dataset + '/train/label',
                                            args.window_size, args.step_size), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(load_function(args.path + args.dataset + '/val/input', args.path + args.dataset + '/val/label',
                                          args.window_size, args.step_size), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(load_function(args.path + args.dataset + '/test/input', args.path + args.dataset + '/test/label',
                                           args.window_size, args.step_size), batch_size=args.batch_size, shuffle=True)
    return train_loader, val_loader, test_loader, ACTIVITIES


def main():
    config = get_args()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_data_loader(config)
    print(config)
    
    _tensor = train_loader.dataset
    _sample_data, _ = _tensor[0]
    if not config.loc_embed:
        config.time_dim, config.sensor_dim = _sample_data.size(-2), _sample_data.size(-1)
    else:
        config.time_dim, config.sensor_dim = _sample_data.size(-2), _sample_data.size(-1) - config.position_indexes
    model = STMoE(config)
    model.apply(init_weights)
    print(model)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    loss_history, val_loss_history, val_f1_history = train(model, optimizer, loss_function, train_loader, val_loader, test_loader, device, 
                                                                                config.epochs)
    targets, outputs = test(model, test_loader, loss_function, device)
    
    if config.plot_train_loss:
        plot_loss(config.dataset, loss_history)
    
    if config.plot_confusion_matrix:
        plot_confusion_matrix(config.dataset, targets, outputs, class_names)
        
    if config.save_model:
        save_model(model, config.save_path, config.dataset)

if __name__ == "__main__":
    main()