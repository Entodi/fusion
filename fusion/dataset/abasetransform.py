import abc
from typing import Tuple, Union

from torch import Tensor


class ABaseTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x) -> Union[Tensor, Tuple[Tensor, ...]]:
        pass
