"""
BSD 3-Clause License

Copyright (c) 2018, Devon Hjelm
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# Modified work Copyright 2020 Alex Fedorov

import torch

from torch import Tensor

from fusion.criterion.mi_estimator import ABaseMIEstimator
from fusion.criterion.mi_estimator.measure import measure_provider


class FenchelDualEstimator(ABaseMIEstimator):
    def __init__(
        self, critic_setting, clip_setting=None, penalty_setting=None, measure="JSD"
    ):
        super().__init__(critic_setting, clip_setting, penalty_setting=penalty_setting)
        self._measure = measure_provider.get(measure, **{})

    def __call__(self, x: Tensor, y: Tensor):
        self._check_input(x, y)

        bs, dim_l, x_locs = x.size()
        _, _, y_locs = y.size()

        scores, penalty = self._compute_scores(x, y)

        pos_mask = torch.eye(bs)
        pos_mask = pos_mask.to(x.device)
        neg_mask = 1 - pos_mask

        e_pos = self._measure.get_positive_expectation(scores)
        e_pos = e_pos.mean(2).mean(2)
        e_pos = (e_pos * pos_mask).sum() / pos_mask.sum()

        e_neg = self._measure.get_negative_expectation(scores)
        e_neg = e_neg.mean(2).mean(2)
        e_neg = (e_neg * neg_mask).sum() / neg_mask.sum()

        loss = e_neg - e_pos
        return loss, penalty
