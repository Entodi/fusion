from typing import Any, Dict, List

import torch.nn as nn
from torch import Tensor

from fusion.model import ABaseModel
from fusion.model.misc import ModelOutput


class Supervised(ABaseModel):
    def __init__(
        self,
        dim_l: int,
        num_classes: int,
        sources: List[int],
        architecture: str,
        architecture_params: Dict[str, Any],
    ):
        """

        Initialization of supervise model

        Args:
            dim_l: output dimension of encoder
            num_classes: number of classes
            architecture: type of architecture
            architecture_params: parameters of architecture

        Returns:
            Supervise model

        """
        super().__init__(sources, architecture, architecture_params)
        assert len(sources) == 1
        self._sources = sources
        self._linear = nn.Linear(dim_l, num_classes, bias=True)
        nn.init.xavier_uniform_(self._linear.weight)
        nn.init.constant_(self._linear.bias, 1 / num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method of supervised models
        
        Args:
            x: input tensor

        Returns:
            result of forward propagation
        """
        assert len(x) == 1
        x = self._source_forward(self._sources[0], x)
        return ModelOutput(z={self._sources[0]: x}, attrs={})

    def _source_forward(self, source_id: int, x: Tensor) -> Tensor:
        x, _ = self._encoder[str(source_id)](x[str(source_id)])
        x = self._linear(x)
        return x
