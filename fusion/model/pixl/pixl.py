from fusion.model.dim import Dim
from fusion.model.misc import ModelOutput

import torch.nn as nn
from torch import Tensor

from typing import Optional, Dict, List, Any, Tuple


class PIXL(Dim):
    def __init__(
        self,
        dim_l: int,
        num_classes: int,
        sources: List[int],
        architecture: str,
        architecture_params: Dict[str, Any],
        conv_head_params: Optional[Dict[str, Any]] = None,
        latent_head_params: Optional[Dict[str, Any]] = None,
        one_source_mode: bool = False
    ):
        super().__init__(
            sources[:1] if one_source_mode else sources,
            architecture,
            architecture_params,
            conv_head_params=conv_head_params,
            latent_head_params=latent_head_params
        )
        self._one_source_mode = one_source_mode
        self._original_sources = sources
        self._linear = nn.ModuleDict()
        for source_id in self._encoder.keys():
            linear = nn.Linear(dim_l, num_classes, bias=True)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 1 / num_classes)
            self._linear[source_id] = linear

    def _source_forward(
        self, source_id: int, x: Tensor
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        if self._one_source_mode:
            main_source = list(self._encoder.keys())[0]
            encoder = self._encoder[main_source]
            linear = self._linear[main_source]
            latent_head = self._latent_heads[main_source]
            conv_head = self._conv_heads[main_source]
        else:
            encoder = self._encoder[source_id]
            linear = self._linear[source_id]
            latent_head = self._latent_heads[source_id]
            conv_head = self._conv_heads[source_id]
        z, latents = encoder(x[source_id])
        # pass latents through projection heads
        #print (latents.keys())
        for key, conv_latent in latents.items():
            conv_latent_size = int(key.split('_')[0])
            if conv_latent_size == 1:
                conv_latent = latent_head(conv_latent)
            elif conv_latent_size > 1:
                #print (conv_latent_size)
                conv_latent = conv_head[key](
                    conv_latent
                )
            else:
                assert False
            latents[key] = conv_latent
        z = linear(z)
        return z, latents

    def forward(self, x: Tensor) -> ModelOutput:
        """

        Args:
            x: input tensor

        Returns:

        """
        ret = ModelOutput(z={}, attrs={})
        ret.attrs["latents"] = {}
        for source_id in x.keys():
            z, conv_latents = self._source_forward(source_id, x)
            ret.z[source_id] = z
            ret.attrs["latents"][source_id] = conv_latents
        return ret

    def get_encoder_list(self):
        if self._one_source_mode:
            main_source = list(self._encoder.keys())[0]
            return {
                source_id: self._encoder[main_source] for source_id in self._original_sources
            }
        else:
            return self._encoder
