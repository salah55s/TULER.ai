import math
from typing import Any
import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from .. import functional as F
from .. import init
from .module import Module


class TULERNN(Module):
    __constants__ = ['in_features', 'out_features']
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TULERNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.threshold = Parameter(torch.empty((out_features, 1), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.threshold)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    def forward(self, input: Tensor) -> Tensor:
        linear_output = F.linear(input, self.weight, self.bias)
        exceed_threshold = linear_output <=  self.threshold.transpose(0, 1)
        output = linear_output * exceed_threshold + linear_output * (~exceed_threshold)
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
class BrainNeuronActivation(Module):
    def __init__(self, num_neurons):
        super(BrainNeuronActivation, self).__init__()
        self.thresholds = Parameter(torch.rand(num_neurons))
        self.parm1 = Parameter(torch.rand(num_neurons))
        self.parm2 = Parameter(torch.rand(num_neurons))
        self.parm3 = Parameter(torch.rand(num_neurons))
        self.parm4 = Parameter(torch.rand(num_neurons))
        self.parm5 = Parameter(torch.rand(num_neurons))

    def forward(self, input):
        output = input.clone()
        output = torch.where(output < 0, self.parm1 * output, output)
        action_potentials = torch.where(output >= self.thresholds, self.parm5 + self.parm2 * (output - self.parm4), self.parm3 * output)
        return action_potentials
