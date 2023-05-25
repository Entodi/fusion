from typing import Dict

import torch.nn as nn
from torch import Tensor

from fusion.model.misc import ModelOutput


class EncoderExtractor(nn.Module):
    def __init__(self, encoder, source_id: int):
        """
        Args:
            encoder:
            num_classes:
            dim_l:
            source_id:
        """
        super().__init__()
        self._encoder = encoder
        self._encoder.eval()
        self._source_id = source_id

    def forward(self, x: Tensor) -> ModelOutput:
        """
        Args:
            x:
            
        Returns:
        """
        x = x[str(self._source_id)]
        x, latents = self._encoder(x)
        x = x.detach()
        if isinstance(latents, (Dict)):
            for k in latents.keys():
                latents[k].detach()
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return ModelOutput(z={self._source_id: x}, attrs={'latents': latents})
