from .base_measure import ABaseMeasure
from .base_measure import (
    GanMeasure,
    JsdMeasure,
    X2Measure,
    KLMeasure,
    RKLMeasure,
    DVMeasure,
    H2Measure,
    W1Measure,
)
from fusion.utils import ObjectProvider


measure_provider = ObjectProvider()
measure_provider.register_object("GAN", GanMeasure)
measure_provider.register_object("JSD", JsdMeasure)
measure_provider.register_object("X2", X2Measure)
measure_provider.register_object("KL", KLMeasure)
measure_provider.register_object("RKL", RKLMeasure)
measure_provider.register_object("DV", DVMeasure)
measure_provider.register_object("H2", H2Measure)
measure_provider.register_object("W1", W1Measure)


__all__ = ["measure_provider", "ABaseMeasure"]
