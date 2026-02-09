import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float):
        assert input.dim() == 2, "SyncBatchNorm expects 2D input (B, C)."
        B, C = input.shape

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        x = input
        local_sum = x.sum(dim=0)
        local_sumsq = (x * x).sum(dim=0)
        local_count = torch.tensor([B], device=x.device, dtype=x.dtype)

        if world_size > 1:
            stats = torch.cat([local_sum, local_sumsq, local_count], dim=0)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_sum = stats[:C]
            global_sumsq = stats[C:2 * C]
            global_count = stats[2 * C]
        else:
            global_sum = local_sum
            global_sumsq = local_sumsq
            global_count = local_count[0]

        mean = global_sum / global_count
        sq_mean = global_sumsq / global_count
        var = sq_mean - mean * mean
        var = torch.clamp(var, min=0.0)
        invstd = torch.rsqrt(var + eps)

        with torch.no_grad():
            running_mean.mul_(1.0 - momentum).add_(momentum * mean.detach())
            if global_count.item() > 1:
                var_unbiased = var * (global_count / (global_count - 1.0))
            else:
                var_unbiased = var

            std_unbiased = torch.sqrt(var_unbiased + eps)
            running_std.mul_(1.0 - momentum).add_(momentum * std_unbiased.detach())
        
        x_hat = (x - mean) * invstd

        ctx.save_for_backward(x, mean, invstd, global_count.to(x.dtype))
        ctx.world_size = world_size
        return x_hat


    @staticmethod
    def backward(ctx, grad_output):
        x, mean, invstd, global_count = ctx.saved_tensors
        world_size = ctx.world_size

        C = mean.numel()

        xmu = x - mean

        local_grad_sum = grad_output.sum(dim=0)
        local_input_grad_sum = (grad_output * xmu).sum(dim=0)

        if world_size > 1:
            stats = torch.cat([local_grad_sum, local_input_grad_sum], dim=0)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_grad_sum = stats[:C]
            global_input_grad_sum = stats[C:]
        else:
            global_grad_sum = local_grad_sum
            global_input_grad_sum = local_input_grad_sum
        
        invstd2 = invstd * invstd
        
        return (invstd / global_count) * (
            global_count * grad_output - global_grad_sum - xmu * invstd2 * global_input_grad_sum
        ), None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        self.register_buffer("running_mean", torch.zeros((num_features,)))
        self.register_buffer("running_std", torch.ones((num_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            return sync_batch_norm.apply(input, self.running_mean, self.running_std, self.eps, self.momentum)
        return (input - self.running_mean) / self.running_std
