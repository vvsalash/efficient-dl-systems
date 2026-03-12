"""
Zero-Centered RMSNorm
"""

import torch
import torch.nn as nn


@torch.compile(fullgraph=True)
def rmsnorm_forward(x, weight, eps):
    """Zero-Centered RMSNorm forward."""
    input_dtype = x.dtype
    x = x.float()
    scale = 1.0 + weight.float()
    rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    output = x * rms_inv * scale
    return output.to(input_dtype), rms_inv


@torch.compile(fullgraph=True)
def rmsnorm_backward(grad_output, x, weight, rms_inv):
    """Zero-Centered RMSNorm backward."""
    x = x.float()
    grad_output = grad_output.float()
    scale = 1.0 + weight.float()
    grad = grad_output * scale

    gx_mean = (grad * x).mean(dim=-1, keepdim=True)
    dx = grad * rms_inv - x * (rms_inv ** 3) * gx_mean
    dweight = (grad_output * (x * rms_inv)).sum(dim=tuple(range(grad_output.ndim - 1)))

    return dx.to(grad_output.dtype), dweight.to(weight.dtype)


class RMSNormFunction(torch.autograd.Function):
    """
    Template for memory-efficient and fused Zero-Centered RMSNorm autograd function.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        output, rms_inv = rmsnorm_forward(x, weight, eps)
        ctx.eps = eps
        ctx.save_for_backward(x, weight, rms_inv)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, rms_inv = ctx.saved_tensors
        dx, dweight = rmsnorm_backward(grad_output, x, weight, rms_inv)
        return dx, dweight, None


class RMSNorm(nn.Module):
    """
    Zero-Centered RMSNorm: y = x/rms(x) * (1 + weight), weight init to zeros.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)
