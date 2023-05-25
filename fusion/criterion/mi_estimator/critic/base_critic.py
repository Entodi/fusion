import abc

from torch import Tensor


class ABaseCritic(abc.ABC):
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        pass
