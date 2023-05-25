from .base_penalty import ABasePenalty
from .l2_penalty import L2Penalty

from fusion.utils import ObjectProvider


penalty_provider = ObjectProvider()
penalty_provider.register_object("L2Penalty", L2Penalty)


__all__ = [
    "ABasePenalty",
    "L2Penalty",
    "penalty_provider",
]
