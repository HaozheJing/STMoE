import torch
from torch import nn
from .Gate import Top2Gating
from inspect import isfunction
from .Experts import Experts


def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val


class MoE(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=16,
                 hidden_dim=None,
                 activation=nn.ReLU,
                 second_policy_train='random',
                 second_policy_eval='random',
                 second_threshold_train=0.2,
                 second_threshold_eval=0.2,
                 capacity_factor_train=1.25,
                 capacity_factor_eval=2.,
                 loss_coef=1e-2,
                 experts=None,
                 dropout=0.2):
        super().__init__()

        self.num_experts = num_experts

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval,
                         'second_threshold_train': second_threshold_train,
                         'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train,
                         'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates=num_experts, **gating_kwargs)
        self.experts = default(experts, lambda: Experts(dim, num_experts=num_experts, hidden_dim=hidden_dim,
                                                        activation=activation, dropout=dropout))
        self.loss_coef = loss_coef
        
        # data_for_visualize
        self.gating_data = None
        self.expert_outputs_data = None
    
    def get_gating_data(self):
        """ 返回门控数据，绘图用 """
        return self.gating_data

    def get_expert_outputs_data(self):
        """ 返回专家网络输出数据，绘图用 """
        return self.expert_outputs_data

    def forward(self, inputs):
        b, n, d, e = *inputs.shape, self.num_experts
        dispatch_tensor, combine_tensor, loss, index_1, index_2 = self.gate(inputs)
        expert_inputs = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor)
        
        self.gating_data = {
            "dispatch_tensor": dispatch_tensor,
            "combine_tensor": combine_tensor,
            "loss": loss,
            "top_experts": {"index_1": index_1, "index_2": index_2}
        }
        
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)
        
        self.expert_outputs_data = expert_outputs
        
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs, combine_tensor)
        return output
