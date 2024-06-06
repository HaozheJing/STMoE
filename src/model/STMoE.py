import torch
from .MOE import MoE
import torch.nn as nn
from .Sensor_pre import SensorPre
from .Input_layer import InputBlocks
from ..hparams import get_args, Config
from .Inputwithposition import InputBlocks2




CONFIG = get_args()

class TimeWeightedSum(nn.Module):
    def __init__(self, time_window):
        super(TimeWeightedSum, self).__init__()
        self.weights = nn.Parameter(torch.linspace(1, 0, steps=time_window))

    def forward(self, x):
        weights = self.weights.unsqueeze(0).unsqueeze(-1) 
        weighted_x = x * weights  
        summed_x = torch.sum(weighted_x, dim=1)  
        return summed_x
    

class STMoE(nn.Module):
    """
    STMoE类实现方法，把前面所有模块进行拼接
    """
    def __init__(self, config: Config):
        super(STMoE, self).__init__()
        self.sensor_preprocessing = SensorPre(config.sensor_dim, config.n_filters, config.kernel_size, config.dilation_rate)
        self.dimension_reducer = nn.Conv1d(in_channels=config.sensor_dim, out_channels =config.out_channels, kernel_size=1)
        
        if config.loc_embed:
            assert config.position_indexes is not None and config.position_dim is not None, "Initialize position_indexes and dimension!"
            self.input_block = InputBlocks2(config.out_channels, config.time_dim, config.position_dim, config.position_indexes, config.sensor_heads, config.time_heads,
                                            config.num_input_layer, config.dropout_rate)
        else:
            self.input_block = InputBlocks(config.out_channels, config.time_dim, config.output_dim, config.sensor_heads, config.time_heads, config.num_input_layer,
                                          config.dropout_rate)
        
        self.moe_block = MoE(config.output_dim, config.num_experts, config.hidden_dim, config.activation, config.second_policy_train, config.second_policy_eval,
                             config.second_threshold_train, config.second_threshold_eval, config.capacity_factor_train, config.capacity_factor_eval, config.loss_coef,
                             config.experts, config.dropout_rate)
        
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.layer_norm = nn.LayerNorm(config.time_dim)
        self.time_weighted_sum = TimeWeightedSum(config.time_dim)
        self.classifier = nn.Linear(config.output_dim, config.num_classes)
    
    def get_gating_data(self):
        """ 从 MoE 模块获取门控数据 """
        return self.moe_block.get_gating_data()

    def get_moe_expert_data(self):
        """ 从 MoE 模块获取专家输出数据 """
        return self.moe_block.get_expert_outputs_data()

    def forward(self, x):
        x, _ = self.sensor_preprocessing(x)
        x = x.transpose(1, 2)
        x = self.dimension_reducer(x)
        x = x.transpose(1, 2)
        
        x = self.input_block(x)
        x = self.moe_block(x)
        #x = self.dropout(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.time_weighted_sum(x)
        x = self.classifier(x)
        x = self.dropout(x)
        
        return x
