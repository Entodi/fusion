# Original work Copyright (c) 2016 Vahid Noroozi
# Modified work Copyright 2019 Zhanghao Wu
# Modified work Copyright 2020 Alex Fedorov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

from fusion.criterion.loss import ABaseLoss
from fusion.criterion.misc.utils import total_loss_summation
from fusion.model.misc import ModelOutput

import torch
from torch import Tensor

from typing import Optional, Tuple, Any, Dict


class CanonicalCorrelation(ABaseLoss):
    def __init__(
        self,
        eps: float = 1e-9, # 1e-3
        r1: float = 1e-3, # 1e-7
        r2: float = 1e-3, # 1e-7
        use_all_singular_values: bool = True,
        num_top_canonical_components: Optional[int] = None,
    ):
        """
        Implementation of the loss functions in the DCCA
        based on mvlearn package
        (https://github.com/mvlearn/mvlearn/blob/main/mvlearn/embed/dcca.py).

        Andrew, Galen, Raman Arora, Jeff Bilmes, and Karen Livescu.
        "Deep canonical correlation analysis."
        In International conference on machine learning,
        pp. 1247-1255. PMLR, 2013.

        Args:
            eps: Parameter for numerical stability
            r1: Parameter for numerical stability
            r2: Parameter for numerical stability
            use_all_singular_values: Boolean flag whether or not to use all the singular values in the loss calculation
            num_top_canonical_components: Number of top canonical components used if use_all_singular_values is false

        Returns:
            Instance of CanonicalCorrelation
        """
        super().__init__()
        self._eps = eps
        self._r1 = r1
        self._r2 = r2
        self._use_all_singular_values = use_all_singular_values
        if not self._use_all_singular_values:
            assert self._num_canonical_components is not None
            assert self._num_canonical_components > 0
        self._num_canonical_components = num_top_canonical_components

    def forward(
        self, preds: ModelOutput, target: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        """
        Forward pass for the loss.

        Args:
            preds: Model output
            target: Targets, however, loss do no use them.

        Returns:
            total_loss: Total loss
            raw_losses: Dictionary for logging with all the computed losses
        """
        del target
        total_loss = None
        raw_losses = {}
        latents = preds.attrs["latents"]
        for source_id_one, l_one in latents.items():
            dim_conv_latent = list(l_one.keys())[-1]
            assert int(dim_conv_latent.split('_')[0]) == 1
            for source_id_two, l_two in latents.items():
                if source_id_one != source_id_two:
                    name = f"CCA_{source_id_one}_{source_id_two}"
                    loss = self._linear_cca(
                        l_one[dim_conv_latent], l_two[dim_conv_latent])
                    raw_losses[name] = loss.item()
                    total_loss = total_loss_summation(total_loss, loss)
        return total_loss, raw_losses

    def _linear_cca(self, h1, h2):
        """
        Computes Linear Canonical Correlation between 2 sources

        Args:
            h1: Representation of the first source
            h2: Representation of the second source
            
        Return:
            Negative correlation between 2 sources
        """
        # Transpose matrices so each column is a sample
        h1, h2 = h1.t(), h2.t()

        o1 = o2 = h1.size(0)
        m = h1.size(1)

        h1_bar = h1 - h1.mean(dim=1).unsqueeze(dim=1)
        h2_bar = h2 - h2.mean(dim=1).unsqueeze(dim=1)
        # Compute covariance matrices and add diagonal so they are
        # positive definite
        sigma_hat12 = (1.0 / (m - 1)) * torch.matmul(h1_bar, h2_bar.t())
        sigma_hat11 = (1.0 / (m - 1)) * torch.matmul(
            h1_bar, h1_bar.t()
        ) + self._r1 * torch.eye(o1, device=h1_bar.device)
        sigma_hat22 = (1.0 / (m - 1)) * torch.matmul(
            h2_bar, h2_bar.t()
        ) + self._r2 * torch.eye(o2, device=h2_bar.device)

        # Calculate the root inverse of covariance matrices by using
        # eigen decomposition
        #[d1, v1] = torch.symeig(sigma_hat11, eigenvectors=True)
        [d1, v1] = torch.linalg.eigh(sigma_hat11, UPLO='U')
        #[d2, v2] = torch.symeig(sigma_hat22, eigenvectors=True)
        [d2, v2] = torch.linalg.eigh(sigma_hat22, UPLO='U')

        # Additional code to increase numerical stability
        pos_ind1 = torch.gt(d1, self._eps).nonzero()[:, 0]
        d1 = d1[pos_ind1]
        v1 = v1[:, pos_ind1]
        pos_ind2 = torch.gt(d2, self._eps).nonzero()[:, 0]
        d2 = d2[pos_ind2]
        v2 = v2[:, pos_ind2]

        # Compute sigma hat matrices using the edited covariance matrices
        sigma_hat11_root_inv = torch.matmul(
            torch.matmul(v1, torch.diag(d1 ** -0.5)), v1.t()
        )
        sigma_hat22_root_inv = torch.matmul(
            torch.matmul(v2, torch.diag(d2 ** -0.5)), v2.t()
        )

        # Compute the T matrix, whose matrix trace norm is the loss
        tval = torch.matmul(
            torch.matmul(sigma_hat11_root_inv, sigma_hat12), sigma_hat22_root_inv
        )

        if self._use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.trace(torch.matmul(tval.t(), tval))
            corr = torch.sqrt(tmp)
        else:
            # just the top self._num_canonical_components singular values are used
            u, v = torch.symeig(torch.matmul(tval.t(), tval), eigenvectors=True)
            u = u.topk(self._num_canonical_components)[0]
            corr = torch.sum(torch.sqrt(u))
        return -corr
