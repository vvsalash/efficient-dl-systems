"""
gpt-oss style SwiGLU Feed-Forward Network

Reference SwiGLU implementation:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings


@triton.jit
def _swiglu_forward_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    stride,
    n_cols: tl.constexpr,
    alpha,
    limit,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)

    gate_ptr += row * stride
    up_ptr += row * stride
    out_ptr += row * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    gate_raw = tl.load(gate_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    up_raw = tl.load(up_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    
    gate = tl.minimum(gate_raw, limit)
    up = tl.maximum(tl.minimum(up_raw, limit), -limit)

    sig = tl.sigmoid(gate * alpha)
    out = (up + 1.0) * gate * sig

    out_dtype = tl.load(out_ptr + col_offsets, mask=mask, other=0).dtype
    tl.store(out_ptr + col_offsets, out.to(out_dtype), mask=mask)


@triton.jit
def _pack_mask_kernel(
    up_ptr,
    mask_ptr,
    stride_up,
    stride_mask,
    n_cols,
    limit,
    BLOCK_BYTES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)

    up_ptr += row * stride_up
    mask_ptr += row * stride_mask

    byte_offsets = block * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    n_mask_bytes = (n_cols + 7) // 8
    valid_bytes = byte_offsets < n_mask_bytes

    bit_offsets = tl.arange(0, 8)
    col_offsets = byte_offsets[:, None] * 8 + bit_offsets[None, :]
    valid_cols = col_offsets < n_cols

    up = tl.load(up_ptr + col_offsets, mask=valid_cols, other=0).to(tl.float32)
    ok = (up >= -limit) & (up <= limit) & valid_cols

    bits = (1 << bit_offsets)[None, :]
    packed = tl.sum(ok.to(tl.int32) * bits, axis=1)

    tl.store(mask_ptr + byte_offsets, packed.to(tl.uint8), mask=valid_bytes)


@triton.jit
def _swiglu_backward_kernel(
    dact_ptr,
    gate_ptr,
    act_ptr,
    mask_ptr,
    stride,
    stride_mask,
    n_cols: tl.constexpr,
    alpha,
    limit,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)

    dact_ptr += row * stride
    gate_ptr += row * stride
    act_ptr += row * stride
    mask_ptr += row * stride_mask

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dact = tl.load(dact_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    gate_raw = tl.load(gate_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    act = tl.load(act_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    gate = tl.minimum(gate_raw, limit)
    sig = tl.sigmoid(gate * alpha)
    glu = gate * sig

    dglu_dgate = sig + gate * alpha * sig * (1.0 - sig)
    gate_ok = gate_raw <= limit

    safe_glu = tl.where(tl.abs(glu) > eps, glu, 1.0)
    up_plus_1 = tl.where(tl.abs(glu) > eps, act / safe_glu, 1.0)

    byte_idx = col_offsets // 8
    bit_idx = col_offsets % 8
    packed = tl.load(mask_ptr + byte_idx, mask=mask, other=0).to(tl.int32)
    up_ok = ((packed >> bit_idx) & 1) != 0

    d_gate = tl.where(gate_ok, dact * up_plus_1 * dglu_dgate, 0.0)
    d_up = tl.where(up_ok, dact * glu, 0.0)

    gate_dtype = tl.load(gate_ptr + col_offsets, mask=mask, other=0).dtype
    act_dtype = tl.load(act_ptr + col_offsets, mask=mask, other=0).dtype

    tl.store(gate_ptr + col_offsets, d_gate.to(gate_dtype), mask=mask)
    tl.store(act_ptr + col_offsets, d_up.to(act_dtype), mask=mask)


def swiglu_forward(
    gate,
    up,
    alpha: float,
    limit: float,
    inplace: bool = False,
):
    shape = gate.shape
    n_cols = shape[-1]
    gate2 = gate.view(-1, n_cols)
    up2 = up.view(-1, n_cols)
    out2 = up2 if inplace else torch.empty_like(up2)

    n_rows = gate2.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_forward_kernel[(n_rows,)](
        gate2,
        up2,
        out2,
        gate2.stride(0),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        alpha=alpha,
        limit=limit,
        num_warps=num_warps,
    )
    return out2.view(*shape)


def _pack_up_mask(up, limit):
    n_cols = up.shape[-1]
    up2 = up.view(-1, n_cols)
    packed = torch.empty((up2.shape[0], (n_cols + 7) // 8), device=up.device, dtype=torch.uint8)

    _pack_mask_kernel[(up2.shape[0], triton.cdiv(packed.shape[1], 16))](
        up2,
        packed,
        up2.stride(0),
        packed.stride(0),
        n_cols,
        limit,
        BLOCK_BYTES=16,
    )
    return packed


def swiglu_backward(
    gate: torch.Tensor,
    act: torch.Tensor,
    packed_mask: torch.Tensor,
    d_activation: torch.Tensor,
    alpha: float,
    limit: float,
):
    shape = gate.shape
    n_cols = shape[-1]
    
    gate2 = gate.view(-1, n_cols)
    act2 = act.view(-1, n_cols)
    dact2 = d_activation.view(-1, n_cols)
    mask2 = packed_mask.view(gate2.shape[0], -1)

    n_rows = gate2.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _swiglu_backward_kernel[(n_rows,)](
        dact2,
        gate2,
        act2,
        mask2,
        gate2.stride(0),
        mask2.stride(0),
        n_cols=n_cols,
        alpha=alpha,
        limit=limit,
        eps=1e-12,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return gate.view(*shape), act.view(*shape)


class MemoryEfficientSwiGLUMLP(torch.autograd.Function):
    """
    Memory-optimized SwiGLU MLP with selective recomputation.
    """
    
    @staticmethod
    def forward(ctx, x, w_gate, w_up, w_down, alpha, limit):
        x2 = x.view(-1, x.size(-1))
        gate = x2 @ w_gate.T
        up = x2 @ w_up.T

        packed_mask = _pack_up_mask(up, limit)
        act = swiglu_forward(gate, up, alpha, limit, inplace=True)
        out = act @ w_down.T

        ctx.x2 = x2
        ctx.save_for_backward(gate, act, packed_mask, w_gate, w_up, w_down)
        ctx.input_shape = x.shape
        ctx.alpha = alpha
        ctx.limit = limit

        return out.view(*x.shape[:-1], w_down.size(0))
    
    @staticmethod
    def backward(ctx, grad_output):
        x2 = ctx.x2
        gate, act, packed_mask, w_gate, w_up, w_down = ctx.saved_tensors

        grad_output2 = grad_output.reshape(-1, grad_output.size(-1))
        grad_w_down = torch.mm(grad_output2.t(), act)
        d_activation = torch.mm(grad_output2, w_down)

        d_gate, d_up = swiglu_backward(
            gate, act, packed_mask, d_activation, ctx.alpha, ctx.limit
        )

        grad_w_gate = torch.mm(d_gate.t(), x2)
        grad_w_up = torch.mm(d_up.t(), x2)

        dx = torch.mm(d_gate, w_gate)
        dx.addmm_(d_up, w_up)
        dx = dx.view(*ctx.input_shape)

        return dx, grad_w_gate, grad_w_up, grad_w_down, None, None


class SwiGLUFeedForward(nn.Module):
    """
    gpt-oss style SwiGLU.
    
    output = W_down @ ((up + 1) * gate * sigmoid(gate * alpha))
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.alpha = 1.702
        self.limit = 7.0

        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MemoryEfficientSwiGLUMLP.apply(
            x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight, self.alpha, self.limit
        )
