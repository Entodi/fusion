from fusion.criterion.loss import ABaseLoss, AE
from fusion.criterion.loss.multi_dim import MultiDim
from fusion.criterion.loss.dim import RR_MODE
from fusion.criterion.misc.utils import total_loss_summation
from fusion.model.misc import ModelOutput
from fusion.utils import Setting

from torch import Tensor
from typing import Optional, Tuple, Any, Dict


class RR_AE(ABaseLoss):
    def __init__(
        self,
        estimator_setting: Setting,
    ):
        # ToDo: Add references to ABaseMIEstimator
        """
        Implementation of the RR-AE loss

        Args:
            estimator_setting: Setting for Mutual Information estimator. See ABaseMIEstimator for details.

        Returns:
            Instance of RR_AE
        """
        super().__init__()
        self._ae_loss = AE()
        self._rr_loss = MultiDim(
            dim_cls=[],
            estimator_setting=estimator_setting,
            modes=[RR_MODE],
            weights=[1.0],
        )

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
        total_loss = None
        raw_losses = {}
        loss, ae_raw = self._ae_loss(preds)
        raw_losses.update(ae_raw)
        total_loss = total_loss_summation(total_loss, loss)
        loss, rr_raw = self._rr_loss(preds)
        raw_losses.update(rr_raw)
        total_loss = total_loss_summation(total_loss, loss)
        return total_loss, raw_losses
