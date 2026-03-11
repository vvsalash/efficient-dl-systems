"""
Cross Entropy Loss for Causal LM
"""

import torch
import torch.nn as nn

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss


class CrossEntropyLoss(nn.Module):
    """Fused Linear Cross Entropy for causal LM."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        shift_hidden = hidden_states[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        shift_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
        shift_labels = shift_labels.view(-1)

        return self.loss_fn(lm_head_weight, shift_hidden, shift_labels)
