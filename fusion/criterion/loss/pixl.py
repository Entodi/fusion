from fusion.criterion.loss import ABaseLoss
from fusion.criterion.loss import CustomCrossEntropyLoss
from fusion.criterion.loss.dim import RR_MODE
from fusion.criterion.loss.multi_dim import MultiDim
from fusion.criterion.misc.utils import total_loss_summation
from fusion.model.misc import ModelOutput
from fusion.utils import Setting

from torch import Tensor

from typing import Optional, List, Dict, Any, Tuple


class PIXL(ABaseLoss):
    def __init__(
        self,
        estimator_setting: List[Setting],
        trade_off: int = 0.5,
    ):
        super().__init__()
        self._trade_off = trade_off
        print ('TRADE OFF', self._trade_off)
        self._rr_loss = MultiDim(
            dim_cls=[],
            estimator_setting=estimator_setting['RR'],
            modes=[RR_MODE],
            weights=[1.0],
        )
        self._ce_loss = CustomCrossEntropyLoss(**estimator_setting['CE'])

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
        loss, ce_raw = self._ce_loss(preds, target)
        loss = self._trade_off * loss
        raw_losses.update(ce_raw)
        total_loss = total_loss_summation(total_loss, loss)
        loss, rr_raw = self._rr_loss(preds)
        loss = (1 - self._trade_off) * loss
        raw_losses.update(rr_raw)
        total_loss = total_loss_summation(total_loss, loss)
        return total_loss, raw_losses
