import abc

from typing import Optional, Tuple, Any, Dict

import torch.nn as nn
from torch import Tensor

from fusion.model.misc import ModelOutput


class ABaseLoss(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(self):
        super().__init__()

    def forward(
        self, preds: ModelOutput, target: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        pass
