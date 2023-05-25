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
# Modified work Copyright 2020 Alex Fedorov

import abc
from typing import Optional, Tuple

from torch import Tensor

from fusion.criterion.mi_estimator.critic import critic_provider
from fusion.criterion.mi_estimator.clip import clip_provider
from fusion.criterion.mi_estimator.penalty import penalty_provider


class ABaseMIEstimator(abc.ABC):
    def __init__(self, critic_setting, clip_setting=None, penalty_setting=None):
        args = {} if critic_setting.args is None else critic_setting.args
        self._critic = critic_provider.get(critic_setting.class_type, **args)
        self._clip = None
        self._penalty = None
        if clip_setting is not None:
            if clip_setting.class_type is not None:
                args = {} if clip_setting.args is None else clip_setting.args
                self._clip = clip_provider.get(clip_setting.class_type, **args)
        if penalty_setting is not None:
            if penalty_setting.class_type is not None:
                args = {} if penalty_setting.args is None else penalty_setting.args
                self._penalty = penalty_provider.get(penalty_setting.class_type, **args)

    @abc.abstractmethod
    def __call__(self, x: Tensor, y: Tensor):
        pass

    def _compute_scores(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        bs, dim_l, x_locs = x.size()
        _, _, y_locs = y.size()

        # bs x dim_l x locations -> bs x locations x dim_l
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)
        # bs*locations x dim_l
        x = x.reshape(-1, dim_l)
        y = y.reshape(-1, dim_l)

        # Outer product bs*xlocs*ylocs x bs*xlocs*ylocs
        scores = self._critic(y, x)
        penalty = None
        if self._penalty is not None:
            penalty = self._penalty(scores)
        if self._clip is not None:
            scores = self._clip(scores)

        # bs*bs*xlocs*ylocs -> bs x y_locs x bs x x_locs
        scores = scores.reshape(bs, y_locs, bs, x_locs)
        # bs x bs x x_locs x y_locs tensor.
        scores = scores.permute(0, 2, 3, 1)
        return scores, penalty

    @staticmethod
    def _check_input(x: Tensor, y: Tensor):
        assert len(x.size()) == 3
        assert len(y.size()) == 3
        assert x.size(0) == y.size(0)
        assert x.size(1) == y.size(1)
