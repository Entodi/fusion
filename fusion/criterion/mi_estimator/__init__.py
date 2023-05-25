from .base_mi_estimator import ABaseMIEstimator
from .fenchel_dual import FenchelDualEstimator
from .infonce import InfoNceEstimator
from fusion.utils import ObjectProvider


mi_estimator_provider = ObjectProvider()
mi_estimator_provider.register_object("FenchelDualEstimator", FenchelDualEstimator)
mi_estimator_provider.register_object("InfoNceEstimator", InfoNceEstimator)


__all__ = [
    "mi_estimator_provider",
    "ABaseMIEstimator",
    "FenchelDualEstimator",
    "InfoNceEstimator",
]
