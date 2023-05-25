import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fusion.architecture.projection_head import ConvHead, LatentHead
from fusion.model import ABaseModel
from fusion.model.misc import ModelOutput


class Dim(ABaseModel):
    def __init__(
        self,
        sources: List[int],
        architecture: str,
        architecture_params: Dict[str, Any],
        conv_head_params: Optional[Dict[str, Any]] = None,
        latent_head_params: Optional[Dict[str, Any]] = None,
    ):
        # create encoders for each source
        super().__init__(sources, architecture, architecture_params)
        assert len(self._sources) in [1, 2]
        self._input_size = architecture_params["input_size"]
        self._input_dim = (
            architecture_params["input_dim"]
            if "input_dim" in architecture_params.keys()
            else 2
        )
        self._latent_head_params = latent_head_params
        # create convolutional heads
        print (architecture_params["dim_cls"])
        self._conv_heads = nn.ModuleDict()
        for source_id in self._encoder.keys():
            self._conv_heads[source_id] = nn.ModuleDict()
            for key in architecture_params["dim_cls"]:
                print (key)
                conv_head_params = self._parse_conv_head_params(
                    conv_head_params, architecture_params, key, source_id
                )
                print (conv_head_params)
                conv_head = ConvHead(**conv_head_params)
                conv_head.init_weights()
                self._conv_heads[str(source_id)][str(key)] = conv_head
        # create latent heads
        self._latent_heads = nn.ModuleDict()
        for source_id in self._encoder.keys():
            latent_head_params = copy.deepcopy(dict(**self._latent_head_params))
            latent_head_params = self._parse_latent_head_params(
                latent_head_params, architecture_params
            )
            latent_head = LatentHead(**latent_head_params)
            latent_head.init_weights()
            self._latent_heads[source_id] = latent_head

    def _source_forward(
        self, source_id: int, x: Tensor
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        z, latents = self._encoder[source_id](x[source_id])
        # pass latents through projection heads
        #print (latents.keys())
        for key, conv_latent in latents.items():
            conv_latent_size = int(key.split('_')[0])
            if conv_latent_size == 1:
                conv_latent = self._latent_heads[source_id](conv_latent)
            elif conv_latent_size > 1:
                #print (conv_latent_size)
                conv_latent = self._conv_heads[source_id][key](
                    conv_latent
                )
            else:
                assert False
            latents[key] = conv_latent
        return z, latents

    def forward(self, x: Tensor) -> ModelOutput:
        """

        Args:
            x: input tensor

        Returns:

        """
        ret = ModelOutput(z={}, attrs={})
        ret.attrs["latents"] = {}
        for source_id, _ in self._encoder.items():
            z, conv_latents = self._source_forward(source_id, x)
            ret.z[source_id] = z
            ret.attrs["latents"][source_id] = conv_latents
        return ret

    def _parse_conv_head_params(
        self,
        conv_head_params: Optional[Dict[str, Any]],
        architecture_params: Dict[str, Any],
        key: int,
        source_id: int,
    ) -> Dict[str, Any]:
        # by design choice
        conv_head_params = copy.deepcopy(dict(**architecture_params))
        conv_head_params.pop("dim_cls")
        conv_head_params.pop("input_size")
        if "pool_layer_class" in conv_head_params.keys():
            conv_head_params.pop("pool_layer_class")
        dim_in = self._find_dim_in(
            key, source_id
        )  # find the dim_in for dim_conv
        conv_head_params["dim_in"] = dim_in
        conv_head_params["dim_h"] = conv_head_params["dim_l"]
        return conv_head_params

    def _parse_latent_head_params(
        self,
        latent_head_params: Optional[Dict[str, Any]],
        architecture_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "dim_in" not in latent_head_params.keys():
            latent_head_params["dim_in"] = architecture_params["dim_l"]
        if 'dim_h' not in latent_head_params.keys():
            latent_head_params["dim_h"] = architecture_params["dim_l"]
        if 'dim_l' not in latent_head_params.keys():
            latent_head_params["dim_l"] = architecture_params["dim_l"]
        return latent_head_params

    def _find_dim_in(self, key, source_id):
        conv_size, layer_id = [int(x) for x in key.split('_')]
        dim_conv = None
        dim_in = self._architecture_params["dim_in"]
        with torch.no_grad():
            batch_size = 2
            if len(dim_in) == 1:
                source_id_int = 0
            elif len(dim_in) < int(source_id) + 1:
                source_id_int = int(source_id) - 1
            else:
                source_id_int = int(source_id)
            dim_in = self._architecture_params["dim_in"][source_id_int]
            dummy_encoder = self._encoder[source_id].eval()
            if self._input_dim == 2:
                dummy_batch = torch.FloatTensor(
                    batch_size, dim_in, self._input_size, self._input_size
                )
            elif self._input_dim == 3:
                dummy_batch = torch.FloatTensor(
                    batch_size,
                    dim_in,
                    self._input_size,
                    self._input_size,
                    self._input_size,
                )
            else:
                raise NotImplementedError
            x = dummy_batch
            for layer_counter, layer in enumerate(dummy_encoder.get_layers()):
                x, conv_latent = layer(x)
                #print (layer_id)
                print (layer_counter, conv_latent.size(-1), conv_size, layer_id)
                if conv_latent.size(-1) == conv_size:
                    if layer_counter == layer_id:
                        dim_conv = conv_latent.size(1)
                        break
            if dim_conv is None:
                assert False, (
                    f"There is no features with "
                    f"convolutional latent size {key} "
                )
        return dim_conv
