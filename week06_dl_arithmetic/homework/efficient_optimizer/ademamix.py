import math
 
import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer


def _scalar_tensor(value: float, device: torch.device) -> Tensor:
    return torch.tensor(float(value), device=device, dtype=torch.float64)

 
def linear_warmup_scheduler(
    step_t: Tensor,
    alpha_end_t: Tensor,
    alpha_start_t: Tensor,
    warmup_t: Tensor
) -> Tensor:
    warmup = torch.clamp(warmup_t, min=1.0)
    a = torch.clamp(step_t / warmup, min=0.0, max=1.0)
    return (1.0 - a) * alpha_start_t + a * alpha_end_t


def linear_hl_warmup_scheduler(
    step_t: Tensor,
    beta_end_t: Tensor,
    beta_start_t: Tensor,
    warmup_t: Tensor
) -> Tensor:
    eps_t = torch.tensor(1e-8, device=step_t.device, dtype=step_t.dtype)
    half_t = torch.tensor(0.5, device=step_t.device, dtype=step_t.dtype)

    def f(beta: Tensor) -> Tensor:
        return torch.log(half_t) / torch.log(beta + eps_t) - 1.0
    
    def f_inv(t: Tensor) -> Tensor:
        return torch.pow(half_t, 1.0 / (t + 1.0))
    
    warmup = torch.clamp(warmup_t, min=1.0)
    a = torch.clamp(step_t / warmup, min=0.0, max=1.0)
    interpolated = (1.0 - a) * f(beta_start_t) + a * f(beta_end_t)
    return f_inv(interpolated)
 
 
@torch.compile(fullgraph=True)
def ademamix_foreach_fn(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avgs_slow: list[Tensor],
    exp_avg_sqs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    eps: float,
    beta1: float,
    beta2: float,
    step_t: Tensor,
    alpha_final_t: Tensor,
    alpha_warmup_t: Tensor,
    beta3_final_t: Tensor,
    beta3_warmup_t: Tensor,
):
    one_t = torch.tensor(1.0, device=step_t.device, dtype=step_t.dtype)
    zero_t = torch.tensor(0.0, device=step_t.device, dtype=step_t.dtype)
    beta1_t = torch.tensor(beta1, device=step_t.device, dtype=step_t.dtype)
    beta2_t = torch.tensor(beta2, device=step_t.device, dtype=step_t.dtype)

    step_t.add_(1.0)

    has_alpha_warmup = alpha_warmup_t > 0.0
    has_beta3_warmup = beta3_warmup_t > 0.0

    alpha_t = torch.where(
        has_alpha_warmup,
        linear_warmup_scheduler(step_t, alpha_final_t, zero_t, alpha_warmup_t),
        alpha_final_t,
    )
    beta3_t = torch.where(
        has_beta3_warmup,
        linear_hl_warmup_scheduler(step_t, beta3_final_t, beta1_t, beta3_warmup_t),
        beta3_final_t,
    )

    bias_correction1_t = one_t - torch.pow(beta1_t, step_t)
    bias_correction2_t = one_t - torch.pow(beta2_t, step_t)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)

    torch._foreach_mul_(exp_avgs_slow, beta3_t)
    grad_slow_scaled = torch._foreach_mul(grads, one_t - beta3_t)
    torch._foreach_add_(exp_avgs_slow, grad_slow_scaled)

    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

    denom = torch._foreach_sqrt(exp_avg_sqs)
    denom = torch._foreach_div(denom, torch.sqrt(bias_correction2_t))
    denom = torch._foreach_add(denom, eps)

    fast_term = torch._foreach_div(exp_avgs, bias_correction1_t)
    slow_term = torch._foreach_mul(exp_avgs_slow, alpha_t)
    update = torch._foreach_add(fast_term, slow_term)
    update = torch._foreach_div(update, denom)

    wd_part = torch._foreach_mul(params, weight_decay)
    update = torch._foreach_add(update, wd_part)

    torch._foreach_add_(params, update, alpha=-lr)
 

class AdEMAMix(Optimizer):
    r"""Implements the AdEMAMix algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999)) 
            corresponding to beta_1, beta_2, beta_3 in AdEMAMix
        alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
        beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay as in AdamW (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=2.0, 
                 beta3_warmup=None, alpha_warmup=None,  eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha, beta3_warmup=beta3_warmup,
                        alpha_warmup=alpha_warmup, weight_decay=weight_decay)
        super(AdEMAMix, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdEMAMix, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            # beta3_warmup = group["beta3_warmup"]
            # alpha_final = group["alpha"]
            # alpha_warmup = group["alpha_warmup"]
 
            params: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs: list[Tensor] = []
            exp_avgs_slow: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []

            first_param = None

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients.")
                
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg_fast"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                
                params.append(p)
                grads.append(grad)
                exp_avgs.append(state["exp_avg_fast"])
                exp_avgs_slow.append(state["exp_avg_slow"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if first_param is None:
                    first_param = p
            
            if not params:
                continue

            device = first_param.device

            compiled_state = group.setdefault("_compiled_state", {})
            if "step_t" not in compiled_state:
                compiled_state["step_t"] = torch.zeros(
                    (), device=device, dtype=torch.float64
                )
            
            step_t = compiled_state["step_t"]

            alpha_warmup = (
                -1.0 if group["alpha_warmup"] is None else float(group["alpha_warmup"])
            )
            beta3_warmup = (
                -1.0 if group["beta3_warmup"] is None else float(group["beta3_warmup"])
            )

            ademamix_foreach_fn(
                params=params,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avgs_slow=exp_avgs_slow,
                exp_avg_sqs=exp_avg_sqs,
                lr=lr,
                weight_decay=weight_decay,
                eps=eps,
                beta1=beta1,
                beta2=beta2,
                step_t=step_t,
                alpha_final_t=_scalar_tensor(group["alpha"], device),
                alpha_warmup_t=_scalar_tensor(alpha_warmup, device),
                beta3_final_t=_scalar_tensor(beta3_final, device),
                beta3_warmup_t=_scalar_tensor(beta3_warmup, device),
            )
        return loss
