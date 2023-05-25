import copy
from typing import Any, Dict, List, Tuple, Optional

from torch import Tensor
import torch.nn as nn

from fusion.architecture.projection_head import LatentHead
from fusion.model import ABaseModel
from fusion.model.misc import ModelOutput


class AE(ABaseModel):
    def __init__(
        self,
        sources: List[int],
        architecture: str,
        architecture_params: Dict[str, Any],
        latent_head_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialization class of autoencoder model

        Args:
            sources:
            architecture: type of architecture
            architecture_params: parameters of architecture

        Return:
            Autoencoder model
        """
        super().__init__(sources, architecture, architecture_params)
        #self._latent_heads = nn.ModuleDict()
        self._latent_head_params = latent_head_params
        self._latent_heads = nn.ModuleDict()
        for source_id in self._encoder.keys():
            latent_head_params = copy.deepcopy(dict(**self._latent_head_params))
            latent_head_params = self._parse_latent_head_params(
                latent_head_params, architecture_params
            )
            print (latent_head_params)
            latent_head = LatentHead(**latent_head_params)
            latent_head.init_weights()
            self._latent_heads[source_id] = latent_head
        print (self._encoder)
        print (self._latent_heads)

    def _parse_latent_head_params(
        self,
        latent_head_params: Optional[Dict[str, Any]],
        architecture_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        # by design choice
        if "dim_in" not in latent_head_params.keys():
            latent_head_params["dim_in"] = architecture_params["dim_l"]
        if 'dim_h' not in latent_head_params.keys():
            latent_head_params["dim_h"] = architecture_params["dim_l"]
        if 'dim_l' not in latent_head_params.keys():
            latent_head_params["dim_l"] = architecture_params["dim_l"]
        return latent_head_params

    def _source_forward(
        self, source_id: int, x: Tensor
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        """
        Args:
            source_id:
            x:

        Returns:
        """
        return self._encoder[source_id](x[source_id])

    def forward(self, x: ModelOutput) -> ModelOutput:
        """
        Forward method of autoencoder model

        Args:

            x: input tensor
            
        Returns:
            Result of forward method of autoencoder model

        """
        ret = ModelOutput(z={}, attrs={})
        ret.attrs["x"] = {}
        ret.attrs["x_hat"] = {}
        ret.attrs["latents"] = {}
        for source_id, _ in self._encoder.items():
            z, x_hat = self._source_forward(source_id, x)
            ret.z[source_id] = z
            if self._latent_heads:
                latent = self._latent_heads[source_id](z)
            else:
                latent = z
            ret.attrs["latents"][source_id] = {"1": latent}
            ret.attrs["x"][source_id] = x[source_id]
            ret.attrs["x_hat"][source_id] = x_hat
        return ret
