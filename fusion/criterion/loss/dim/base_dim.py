import abc
from typing import Optional

from fusion.criterion.mi_estimator import ABaseMIEstimator
from fusion.criterion.misc.utils import total_loss_summation


class BaseDim(abc.ABC):
    _name: Optional[str] = None

    def __init__(
        self,
        estimator: ABaseMIEstimator,
        weight: float = 1.0,
    ):
        self._estimator = estimator
        self._weight = weight

    @abc.abstractmethod
    def __call__(self, reps, convs):
        pass

    def _update_loss(self, name, total_loss, raw_losses, loss, penalty):
        loss = self._weight * loss
        raw_losses[f"{name}_loss"] = loss.item()
        total_loss = total_loss_summation(total_loss, loss)
        if penalty is not None:
            raw_losses[f"{name}_penalty"] = penalty.item()
            total_loss = total_loss_summation(total_loss, penalty)
        return total_loss, raw_losses
