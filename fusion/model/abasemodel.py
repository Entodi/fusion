import abc
import copy
from typing import Any, Dict, List

import torch.nn as nn
from torch import Tensor

from fusion.architecture import architecture_provider


class ABaseModel(abc.ABC, nn.Module):
    @abc.abstractmethod
    def __init__(
        self, sources: List[int], architecture: str, architecture_params: Dict[str, Any]
    ):
        """
        Args:
            sources:
            architecture:
            architecture_params:
        """
        super().__init__()
        self._architecture = architecture
        self._architecture_params = architecture_params
        self._sources = sources
        self._encoder = nn.ModuleDict({})
        for i, source_id in enumerate(self._sources):
            new_architecture_params = copy.deepcopy(architecture_params)
            new_architecture_params["dim_in"] = architecture_params["dim_in"][i]
            encoder = architecture_provider.get(
                architecture, **new_architecture_params
            )
            encoder.init_weights()
            self._encoder[str(source_id)] = encoder

    @abc.abstractmethod
    def _source_forward(self, source_id: int, x: Tensor) -> Any:
        pass

    def get_encoder(self, source_id: int = 0):
        assert str(source_id) in self._encoder.keys()
        return self._encoder[str(source_id)]

    def get_encoder_list(self):
        return self._encoder
