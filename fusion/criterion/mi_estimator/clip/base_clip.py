import abc

from torch import Tensor


class ABaseClip(abc.ABC):
    def __init__(self, clip_value: float = 10.0):
        self._clip_value = clip_value

    def __call__(self, scores: Tensor) -> Tensor:
        pass

    @property
    def clip_value(self) -> float:
        return self._clip_value

    @clip_value.setter
    def clip_value(self, value: float):
        self._clip_value = value
