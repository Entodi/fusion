"""
MIT License

Copyright (c) [2019] [Philip Bachman]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Optional, Tuple, Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from fusion.criterion.loss import ABaseLoss
from fusion.criterion.loss.dim import dim_mode_provider
from fusion.criterion.loss.dim import CR_MODE, RR_MODE, CC_MODE, XX_MODE
from fusion.criterion.mi_estimator import mi_estimator_provider
from fusion.criterion.misc.utils import total_loss_summation
from fusion.model.misc import ModelOutput
from fusion.utils import Setting


class MultiDim(ABaseLoss):
    def __init__(
        self,
        dim_cls: List[int],
        estimator_setting: Setting,
        modes: List[str] = [CR_MODE, XX_MODE, CC_MODE, RR_MODE],
        weights: List[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        super().__init__()
        assert len(modes) == len(weights)
        self._dim_cls = dim_cls
        self._modes = modes
        self._masks = self._create_masks()
        self._estimator = mi_estimator_provider.get(
            estimator_setting.class_type, **estimator_setting.args
        )
        self._objectives = {}
        for i, mode in enumerate(modes):
            dim_mode_args = {
                "estimator": self._estimator,
                "weight": weights[i],
            }
            self._objectives[mode] = dim_mode_provider.get(mode, **dim_mode_args)

    def _create_masks(self) -> Dict[int, nn.parameter.Parameter]:
        self._masks = {}

    @staticmethod
    def _reshape_target(target: Tensor) -> Tensor:
        return target.reshape(target.size(0), target.size(1), -1)

    @staticmethod
    def _sample_location(
        conv_latents: Tensor, mask: Optional[nn.parameter.Parameter]
    ) -> Tensor:
        n_batch = conv_latents.size(0)
        n_channels = conv_latents.size(1)
        if mask is not None:
            # subsample from conv-ish r_cnv to get a single vector
            mask_idx = torch.randint(0, mask.size(0), (n_batch,))
            if torch.cuda.is_available():
                mask_idx = mask_idx.cuda(torch.device("cuda:{}".format(0)))
                mask = mask.cuda()
            conv_latents = torch.masked_select(conv_latents, mask[mask_idx])
        # flatten features for use as globals in glb->lcl nce cost
        locations = conv_latents.reshape(n_batch, n_channels, 1)
        return locations

    def _prepare_reps_convs(self, latents: Dict[int, Dict[int, Tensor]]):
        reps: Dict[int, Dict[int, Tensor]] = {}
        convs: Dict[int, Dict[int, Tensor]] = {}
        for source_id in latents.keys():
            reps[source_id] = {}
            convs[source_id] = {}
            for key in latents[source_id].keys():
                conv_latent_size = int(key.split('_')[0])
                if conv_latent_size == 1:
                    source = self._sample_location(
                        latents[source_id][key], mask=None
                    )
                    reps[source_id][key] = source
                elif conv_latent_size > 1:
                    source = self._sample_location(
                        latents[source_id][key],
                        mask=self._masks[conv_latent_size],
                    )
                    reps[source_id][key] = source
                    target = self._reshape_target(latents[source_id][key])
                    convs[source_id][key] = target
                else:
                    assert conv_latent_size < 0
        return reps, convs

    def forward(
        self, preds: ModelOutput, target: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        del target
        # prepare sources and targets
        latents = preds.attrs["latents"]
        reps, convs = self._prepare_reps_convs(latents)
        # compute losses
        total_loss = None
        raw_losses = {}
        for _, objective in self._objectives.items():
            loss, raw = objective(reps, convs)
            total_loss = total_loss_summation(total_loss, loss)
            raw_losses.update(raw)
        return total_loss, raw_losses


class SpatialMultiDim(MultiDim):
    def _create_masks(self) -> Dict[int, nn.parameter.Parameter]:
        masks = {}
        for dim_cl in self._dim_cls:
            dim_cl, _ = [int(x) for x in dim_cl.split('_')]
            mask = np.zeros((dim_cl, dim_cl, 1, dim_cl, dim_cl))
            for i in range(dim_cl):
                for j in range(dim_cl):
                    mask[i, j, 0, i, j] = 1
            mask = torch.BoolTensor(mask)
            mask = mask.reshape(-1, 1, dim_cl, dim_cl)
            masks[dim_cl] = nn.Parameter(mask, requires_grad=False)
            if torch.cuda.is_available():
                masks[dim_cl].cuda()
        return masks


class VolumetricMultiDim(MultiDim):
    def _create_masks(self) -> Dict[int, nn.parameter.Parameter]:
        masks = {}
        for dim_cl in self._dim_cls:
            dim_cl, _ = [int(x) for x in dim_cl.split('_')]
            mask = np.zeros((dim_cl, dim_cl, dim_cl, 1, dim_cl, dim_cl, dim_cl))
            for i in range(dim_cl):
                for j in range(dim_cl):
                    for k in range(dim_cl):
                        mask[i, j, k, 0, i, j, k] = 1
            mask = torch.BoolTensor(mask)
            mask = mask.reshape(-1, 1, dim_cl, dim_cl, dim_cl)
            masks[dim_cl] = nn.Parameter(mask, requires_grad=False)
            if torch.cuda.is_available():
                masks[dim_cl].cuda()
        return masks
