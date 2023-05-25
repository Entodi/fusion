import abc

from torch import Tensor


class ABasePenalty(abc.ABC):
    def __call__(self, scores: Tensor) -> Tensor:
        pass
