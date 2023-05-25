from typing import Optional, Tuple, Any, Dict

import torch.nn as nn
from torch import Tensor

from fusion.criterion.misc.utils import total_loss_summation
from fusion.model.misc import ModelOutput

from . import ABaseLoss


class AE(ABaseLoss):
    def __init__(self, **kwargs):
        """
        Args:
            kwargs:
        """
        super().__init__()
        self._loss = nn.MSELoss(**kwargs)

    def forward(
        self, preds: ModelOutput, target: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        """
        Args:
            preds:
            target:
        
        Returns:
        
        """
        total_loss = None
        raw_losses = {}
        for source_id in preds.z.keys():
            x_hat = preds.attrs["x_hat"][source_id]
            x = preds.attrs["x"][source_id]
            loss = self._loss(x_hat, x)
            total_loss = total_loss_summation(total_loss, loss)
            name = f"AE_{source_id}"
            raw_losses[name] = loss.item()
        return total_loss, raw_losses
