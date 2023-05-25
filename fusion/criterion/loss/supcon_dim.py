# Modified work Copyright 2020 Alex Fedorov
"""
BSD 2-Clause License

Copyright (c) 2020, Yonglong Tian
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

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


from __future__ import print_function

import torch
import torch.nn as nn
from torch import Tensor

from fusion.criterion.loss import ABaseLoss
from fusion.model.misc import ModelOutput
from fusion.criterion.mi_estimator.critic import critic_provider
from fusion.criterion.mi_estimator.clip import clip_provider
from fusion.criterion.mi_estimator.penalty import penalty_provider

from typing import Optional, List, Dict, Any, Tuple


class SupConDim(ABaseLoss):
    def __init__(
        self,
        critic_setting,
        temperature=1,
        contrast_mode='all',
        base_temperature=1,
        clip_setting=None,
        penalty_setting=None
    ):
        super().__init__()
        self._loss = SupConLoss(
            critic_setting=critic_setting,
            temperature=temperature,
            contrast_mode=contrast_mode,
            base_temperature=base_temperature,
            clip_setting=clip_setting,
            penalty_setting=penalty_setting
        )

    def forward(
        self, preds: ModelOutput, target: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        features = None
        # preds.z to [bsz, n_views, ...] format
        for source_id, z in preds.z.items():
            if features is not None:
                features = torch.cat([features, z.unsqueeze(1)], dim=1)
            else:
                features = z.unsqueeze(1)
        total_loss, penalty = self._loss(features, labels=target)
        raw_losses = {'SupCon': total_loss}
        if penalty is not None:
            total_loss += penalty
            raw_losses['penalty'] = penalty
        #assert not torch.isnan(total_loss)
        return total_loss, raw_losses


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(
        self,
        critic_setting,
        temperature=1,
        contrast_mode='all',
        base_temperature=1,
        clip_setting=None,
        penalty_setting=None
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
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
                self._penalty = penalty_provider.get(
                    penalty_setting.class_type, **args)

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
                
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        scores = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        scores = self._critic(anchor_feature, contrast_feature)
        scores = torch.div(scores, self.temperature)
        penalty = None
        if self._penalty is not None:
            penalty = self._penalty(scores)
        if self._clip is not None:
            scores = self._clip(scores)

        # for numerical stability
        logits_max, _ = torch.max(scores, dim=1, keepdim=True)
        logits = scores - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        if torch.isnan(loss):
            print (labels)
            assert False
        return loss, penalty
