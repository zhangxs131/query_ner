import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Optional
from torch.nn.modules.loss import _WeightedLoss


class RDropCrossEntropyLoss(_WeightedLoss):
    r"""
    """
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = 'mean',
        rdrop_alpha: float = 1.0
    ) -> None:
        super(RDropCrossEntropyLoss, self).__init__(
            weight,
            size_average,
            reduce,
            reduction
        )
        self.ignore_index = ignore_index
        self.rdrop_alpha = rdrop_alpha

    @staticmethod
    def multilabel_categorical_crossentropy(y_true, y_pred):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return neg_loss + pos_loss

    def compute_kl_loss(self, p, q):

        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1),
            F.softmax(q, dim=-1),
            reduction='none'
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1),
            F.softmax(p, dim=-1),
            reduction='none'
        )

        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        return (p_loss + q_loss) / 2

    def forward(
        self,
        input_a: Tensor,
        input_b: Tensor,
        target: Tensor
    ) -> Tensor:

        bh = input_a.shape[0] * input_a.shape[1]
        target = torch.reshape(target, (bh, -1))

        input_a = torch.reshape(input_a, (bh, -1))
        input_b = torch.reshape(input_b, (bh, -1))

        ce_loss_a = self.multilabel_categorical_crossentropy(
            input_a,
            target,
        )

        ce_loss_b = self.multilabel_categorical_crossentropy(
            input_b,
            target,
        )

        ce_loss = 0.5 * (ce_loss_a + ce_loss_b)

        kl_loss = self.compute_kl_loss(input_a, input_b)

        return ce_loss + self.rdrop_alpha * kl_loss