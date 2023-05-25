from typing import Tuple, Optional, Dict, Any, List

from fusion.criterion.loss import ABaseLoss
from fusion.criterion.loss.dim import CR_MODE
from fusion.criterion.loss.multi_dim import SpatialMultiDim, VolumetricMultiDim
from fusion.criterion.misc.utils import total_loss_summation
from fusion.criterion.misc import CanonicalCorrelation
from fusion.model.misc import ModelOutput
from fusion.utils import Setting

from torch import Tensor


def choose_multi_dim(input_dim):
    """
    Chooses implementation for Multi Dim objective based on a number of the input dimensions

    Args:
        input_dim: The number of input dimensions, e.g. an image is 2-dimensional (input_dim=2) and 
        a volume is 3-dimensional (input_dim=3)
        
    Returns:
        multi_dim_type:
    """
    if input_dim == 2:
        multi_dim_type = SpatialMultiDim
    elif input_dim == 3:
        multi_dim_type = VolumetricMultiDim
    else:
        raise NotImplementedError
    return multi_dim_type


class CR_CCA(ABaseLoss):
    def __init__(
        self,
        dim_cls: List[int],
        estimator_setting: Setting,
        cca_args: Dict[str, Any] = {},
        input_dim: int = 2,
    ):
        # ToDo: Add references to CanonicalCorrelation
        """
        Implementation of the CR-CCA loss

        Args:
            dim_cls: A list of scalars, where each number should correspond to the output width for one of the 
            convolutional layers.
                            The information between latent variable z and the convolutional feature maps width 
                            widths in dim_cls are maximized.
            estimator_setting: Setting for Mutual Information estimator. See ABaseMIEstimator for details.
            cca_args: See CanonicalCorrelation
            input_dim: The number of input dimensions, e.g. an image is 2-dimensional (input_dim=2) and a volume 
            is 3-dimensional (input_dim=3), default=2
            
        Returns:
            Instance of CR_CCA
        """
        super().__init__()
        assert len(dim_cls) > 0, "CR requires at leas one convolutional feature size"
        self._cr_loss = choose_multi_dim(input_dim)(
            dim_cls, estimator_setting=estimator_setting, modes=[CR_MODE], weights=[1.0]
        )
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
        loss, cr_raw = self._cr_loss(preds)
        raw_losses.update(cr_raw)
        total_loss = total_loss_summation(total_loss, loss)
        loss, cca_raw = self._cca_loss(preds)
        raw_losses.update(cca_raw)
        total_loss = total_loss_summation(total_loss, loss)
        return total_loss, raw_losses
