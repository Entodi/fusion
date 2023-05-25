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
# Modified work Copyright 2020 Alex Fedorov


from typing import Optional, Tuple

import torch
from torch import Tensor

from fusion.criterion.mi_estimator import ABaseMIEstimator


class InfoNceEstimator(ABaseMIEstimator):
    def __call__(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        assert self._clip is not None
        self._check_input(x, y)
        bs, dim_l, x_locs = x.size()
        _, _, y_locs = y.size()
        scores, penalty = self._compute_scores(x, y)
        # bs x bs
        pos_mask = torch.eye(bs)
        # bs x bs x 1
        pos_mask = pos_mask.unsqueeze(2)
        # bs x bs x x_locs
        pos_mask = pos_mask.expand(-1, -1, x_locs)
        # bs x bs x x_locs x 1
        pos_mask = pos_mask.unsqueeze(3)
        # bs x bs x x_locs x y_locs
        pos_mask = pos_mask.expand(-1, -1, -1, y_locs).float()
        pos_mask = pos_mask.to(x.device)
        # bs x bs x x_locs x y_locs
        pos_scores = pos_mask * scores
        pos_scores = pos_scores.reshape(bs, bs, -1)
        pos_scores = pos_scores.sum(1)
        neg_mask = 1 - pos_mask
        # bs x bs x x_locs x y_locs
        neg_scores = neg_mask * scores
        # mask self-examples
        # neg_scores -= 10 * pos_mask
        neg_scores -= self._clip.clip_value * pos_mask
        neg_scores = neg_scores.reshape(bs, -1)
        neg_mask = neg_mask.reshape(bs, -1)
        neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]
        neg_sumexp = (neg_mask * torch.exp(neg_scores - neg_maxes)).sum(
            dim=1, keepdim=True
        )
        all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)
        pos_shiftexp = pos_scores - neg_maxes
        nce_scores = pos_shiftexp - all_logsumexp
        nce_scores = -nce_scores.mean()
        return nce_scores, penalty
