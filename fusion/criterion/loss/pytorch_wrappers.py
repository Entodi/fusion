from typing import Optional, Tuple, Any, Dict

from fusion.criterion.misc.utils import total_loss_summation
from fusion.model.misc import ModelOutput

import torch
import torch.nn as nn
from torch import Tensor

from . import ABaseLoss


class CustomCrossEntropyLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """
        Initilization of pytorch wrapper of class Cross Entropy Loss

        Args:
            kwargs: parameters of Cross Entropy Loss

        Returns:
            Class of Cross Entropy Loss
        """
        super().__init__()
        self._loss = nn.CrossEntropyLoss(**kwargs)

    def forward(
        self, preds: ModelOutput, target: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        """
        Forward method of class Cross Entropy Loss

        Args:
            preds: input model's output
            target: target tensor

        Returns:
            Cross Entropy Loss between input and target tensor
        """
        total_loss = None
        raw_losses = {}
        for source_id, z in preds.z.items():
            loss = self._loss(z, target)
            total_loss = total_loss_summation(total_loss, loss)
            raw_losses[f"CE{source_id}"] = loss
        return total_loss, raw_losses

class BCEWithLogitsLoss(ABaseLoss):
    def __init__(self, **kwargs):
        """
        Initilization of pytorch wrapper of class Binary Cross Entropy with
        logits loss
        
        Args:
            kwargs: parameters of Cross Entropy Loss

        Returns:
            Class of Binary Cross Entropy with logits loss
        """
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(**kwargs)

    def forward(
        self, preds: Tensor, target: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        """
        Forward method of class Binary Cross Entropy with
        logits loss

        Args:
            preds: input tensor
            target: target tensor

        Returns:
             Class of Binary Cross Entropy with logits loss
             between input and target tensor
        """
        assert target is not None
        total_loss = None
        raw_losses = {}
        for source_id, z in preds.z.items():
            labels = torch.nn.functional.one_hot(target, num_classes=z.shape[1]).float()
            loss = self._loss(z.squeeze(1), labels.float())
            total_loss = total_loss_summation(total_loss, loss)
            raw_losses[f"BCE{source_id}"] = loss
        return total_loss, raw_losses
