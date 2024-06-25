import torch
from torch import nn
from .int8_matmul import int8_matmul


class Int8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
    
    def forward(self, x):
        out = int8_matmul(x, self.weight.t())
        if self.bias is not None:
            out += self.bias
        return out
