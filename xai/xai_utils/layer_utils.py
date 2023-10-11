import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional, Tuple, Dict
from torch import nn, Tensor
import math


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LayerNormXAI(nn.Module):
    __constants__ = ['weight', 'bias', 'eps']

    def __init__(self, hidden, eps=1e-5, elementwise_affine=True, args=None, dtype=None):
                
        factory_kwargs = {'device': DEVICE, 'dtype': dtype}
        super(LayerNormXAI, self).__init__()
        self.mode = args.lnv
        self.sigma = args.sigma
        self.hidden = hidden
        self.adanorm_scale = args.adanorm_scale
        self.nowb_scale = args.nowb_scale
        self.mean_detach = args.mean_detach
        self.std_detach = args.std_detach
        if self.mode == 'no_norm':
            elementwise_affine = False
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.hidden, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.hidden, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.mode == 'no_norm':
            return input
        elif self.mode == 'topk':
            T, B, C = input.size()
            input = input.reshape(T * B, C)
            k = max(int(self.hidden * self.sigma), 1)
            input = input.view(1, -1, self.hidden)
            topk_value, topk_index = input.topk(k, dim=-1)
            topk_min_value, top_min_index = input.topk(k, dim=-1, largest=False)
            top_value = topk_value[:, :, -1:]
            top_min_value = topk_min_value[:, :, -1:]
            d0 = torch.arange(top_value.shape[0], dtype=torch.int64)[:, None, None]
            d1 = torch.arange(top_value.shape[1], dtype=torch.int64)[None, :, None]
            input[d0, d1, topk_index] = top_value
            input[d0, d1, top_min_index] = top_min_value
            input = input.reshape(T, B, self.hidden)
            return F.layer_norm(
                input, torch.Size([self.hidden]), self.weight, self.bias, self.eps)
        elif self.mode == 'adanorm':
            mean = input.mean(-1, keepdim=True)
            std = input.std(-1, keepdim=True)
            input = input - mean
            mean = input.mean(-1, keepdim=True)
            graNorm = (1 / 10 * (input - mean) / (std + self.eps)).detach()
            input_norm = (input - input * graNorm) / (std + self.eps)
            return input_norm * self.adanorm_scale
        elif self.mode == 'nowb':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)
            return input_norm

        elif self.mode == 'distillnorm' or self.mode == 'bertnorm':
            mean = input.mean(dim=-1, keepdim=True)
            std = torch.std(input, dim=-1, keepdim=True, unbiased=False) # unbiased deactivates Bessel's correction
            std_real = torch.sqrt(((input - mean) ** 2).sum(dim=-1, keepdims=True) / input.shape[-1])
            if not torch.all(std.eq(std_real)):
                logging.debug("STD calculation if off!")
            std = torch.sqrt(((input - mean) ** 2).sum(dim=-1, keepdims=True) / input.shape[-1])

            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)

            input_norm = input_norm * self.weight + self.bias

            return input_norm

        elif self.mode == 'gradnorm':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input - mean) / (std + self.eps)
            output = input.detach() + input_norm
            return output
        
        else:
            raise


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False, args=None):
    if args is not None:
        if args.lnv != 'origin':
            return LayerNormImpl(hidden=normalized_shape, args=args, eps=eps, elementwise_affine=elementwise_affine)
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
        
        
        
class NewGELUActivationXAI(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, input: Tensor) -> Tensor:   
        # x = x*(act_func(x)/x).detach()     
        func = 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        output = input*((func/(input+1e-6)).detach())
        return  output

    
class GELUActivationXAI(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        
        func = self.act(input)
        return input*((func/(input+1e-6)).detach())
        
        
        
        
class ActivationXAI(nn.Module):
    """
    General implementaton for conservative gradient computation in Transformer nonlinear activation functions.
        References:
    See https://proceedings.mlr.press/v162/ali22a.html
    """
    def __init__(self, act_func):
        super().__init__()
        self.act = act_func

    def forward(self, input: Tensor) -> Tensor:
        
        func = self.act(input)
        return input*((func/(input+1e-6)).detach())
        
        
    
    