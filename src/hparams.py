from dataclasses import dataclass, field
import torch.nn as nn
import argparse


@dataclass
class Config:
    num_classes: int
    dataset: str
    sensor_dim: int
    time_dim: int
    plot_confusion_matrix: bool
    plot_train_loss: bool
    save_model: bool
    save_path: str
    path: str = 'data/'
    seed: int = 32
    batch_size: int = 128
    window_size: int = 128
    step_size: int = 1
    epochs: int = 24
    lr: float = 0.005
    num_input_layer: int = 1
    num_experts: int = 4
    out_channels: int = 128
    output_dim: int = 48
    sensor_heads: int = 4
    time_heads: int = 4
    n_filters: int = 128
    kernel_size: int = 3
    dilation_rate: int = 2
    hidden_dim: int = field(default=None)
    position_dim: int = field(default=None)
    position_indexes: int = field(default=None)
    loc_embed: bool = False
    activation: str = 'ReLU'
    second_policy_train: str = 'random'
    second_policy_eval: str = 'random'
    second_threshold_train: float = 0.2
    second_threshold_eval: float = 0.2
    capacity_factor_train: float = 1.25
    capacity_factor_eval: float = 2.0
    loss_coef: float = 1e-2
    experts: str = field(default=None)
    dropout_rate: float = 0.2


def get_args() -> Config:
    parser = argparse.ArgumentParser(description="Run the STMoE model for activity recognition.")
    # hyper-parameters
    parser.add_argument('--epochs', type=int, default=24, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training and testing.')
    parser.add_argument('--window_size', type=int, default=128, help='Window size for the input features.')
    parser.add_argument('--step_size', type=int, default=1, help='Step size for the sliding window operation.')
    parser.add_argument('--seed', type=int, default=32, help='Random seed for reproducibility.')
    
    # basic-parameters
    parser.add_argument('--plot_train_loss', action='store_true', help='Save training loss plot in save folder')
    parser.add_argument('--plot_confusion_matrix', action='store_true', help='Save confusion matrix plot in save folder')
    parser.add_argument('--save_model', action='store_true', help='Save model parameters')
    parser.add_argument('--save_path', type=str, default='save/model', help='Path to save model parameters')

    # data-parameters
    parser.add_argument('--dataset', type=str, required=True, choices=['UCI', 'OPP', 'HAD'],
                        help='Dataset name to use.')
    parser.add_argument('--path', type=str, default='data/',
                        help='Path for dataset. Please put your data in the data floder and create three folders neamed \
                        train, val and test.Each folder should include 2 folders named input and label')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes for classification.')

    # model-parameters
    parser.add_argument('--sensor_dim', type=int, help='Dimension of sensor data input.')
    parser.add_argument('--time_dim', type=int, help='Temporal dimension of the input data.')
    parser.add_argument('--num_input_layer', type=int, default=1,
                        help='Number of encoder layers of sensor and time attention.')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in the Mixture of Experts layer.')
    parser.add_argument('--out_channels', type=int, default=128,
                        help='Number of output channels in the convolutional layers.')
    parser.add_argument('--output_dim', type=int, default=48,
                        help='Dimensionality of the output from the input blocks.')
    parser.add_argument('--sensor_heads', type=int, default=4,
                        help='Number of heads in the multi-head attention mechanism for sensor data.')
    parser.add_argument('--time_heads', type=int, default=4,
                        help='Number of heads in the multi-head attention mechanism for time data.')
    parser.add_argument('--n_filters', type=int, default=128,
                        help='Number of filters in the SensorPre preprocessing layer.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for convolution operations.')
    parser.add_argument('--dilation_rate', type=int, default=2,
                        help='Dilation rate for dilated convolutions in SensorPre.')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension for experts in the Mixture of Experts layer.')
    parser.add_argument('--position_dim', type=int, default=None,
                        help='Dimension of positional embeddings if loc_embed is enabled.')
    parser.add_argument('--position_indexes', type=int, default=None, nargs='*',
                        help='Indexes for positional encoding.')
    parser.add_argument('--loc_embed', default=False, action='store_true', help='Flag to enable position embeddings.')
    parser.add_argument('--activation', type=eval, default=nn.ReLU, help='Activation function to use in the MoE layer.')
    parser.add_argument('--second_policy_train', type=str, default='random',
                        help='Policy for secondary expert selection during training.')
    parser.add_argument('--second_policy_eval', type=str, default='random',
                        help='Policy for secondary expert selection during evaluation.')
    parser.add_argument('--second_threshold_train', type=float, default=0.2,
                        help='Threshold for secondary expert training policy.')
    parser.add_argument('--second_threshold_eval', type=float, default=0.2,
                        help='Threshold for secondary expert evaluation policy.')
    parser.add_argument('--capacity_factor_train', type=float, default=1.25,
                        help='Capacity factor during training for the MoE layer.')
    parser.add_argument('--capacity_factor_eval', type=float, default=2.0,
                        help='Capacity factor during evaluation for the MoE layer.')
    parser.add_argument('--loss_coef', type=float, default=1e-2,
                        help='Coefficient for the loss term related to MoE expert load balancing.')
    parser.add_argument('--experts', type=str, default=None, help='Type of experts to use in the MoE layer.')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate applied in various layers of the model.')

    args = parser.parse_args()
    return Config(**vars(args))