from fusion.criterion.loss import ABaseLoss, AE
from fusion.criterion.misc import CanonicalCorrelation
from fusion.criterion.misc.utils import total_loss_summation
from fusion.model.misc import ModelOutput

from torch import Tensor
from typing import Optional, Tuple, Any, Dict


class DCCAE(ABaseLoss):
    def __init__(
        self,
        cca_args: Dict[str, Any] = {},
    ):
        # ToDo: Add references to CanonicalCorrelation
        """
        Implementation of the DCCAE loss

        Args:
            cca_args: See CanonicalCorrelation

        Returns:
            Instance of DCCAE
        """
        super().__init__()
        self._ae_loss = AE()
        self._cca_loss = CanonicalCorrelation(**cca_args)

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
        loss, cca_raw = self._cca_loss(preds)
        raw_losses.update(cca_raw)
        total_loss = total_loss_summation(total_loss, loss)
        return total_loss, raw_losses
