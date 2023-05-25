import logging

import torch
import torch.nn as nn
from torch import Tensor

from fusion.architecture.base_block import Flatten
from fusion.model.misc import ModelOutput


class LinearEvaluator(nn.Module):
    def __init__(self, encoder, num_classes: int, dim_l: int, source_id: int, freeze: bool = True):
        """
        Args:
            encoder:
            num_classes:
            dim_l:
            source_id:
        """
        super().__init__()
        self._encoder = encoder
        if freeze:
            self._encoder.eval()
        self._flatten = Flatten()
        self._linear = nn.Linear(dim_l, num_classes, bias=True)
        nn.init.xavier_uniform_(self._linear.weight)
        nn.init.constant_(self._linear.bias, 1 / num_classes)
        self._source_id = source_id
        self._freeze = freeze
        logging.info(f'Freeze: {self._freeze}')

    def _encoder_forward(self, x):
        x = self._encoder(x)[0]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self._flatten(x)
        return x

    def forward(self, x: Tensor) -> ModelOutput:
        """
        Args:
            x:
            
        Returns:
        """
        x = x[str(self._source_id)]
        if self._freeze:
            with torch.no_grad():
                x = self._encoder_forward(x)
                x = x.detach()
        else:
            x = self._encoder_forward(x)
        x = self._linear(x)
        return ModelOutput(z={self._source_id: x}, attrs={})
