from .base_critic import ABaseCritic
from .separable_critic import SeparableCritic, ScaledDotProduct, CosineSimilarity

from fusion.utils import ObjectProvider

critic_provider = ObjectProvider()
critic_provider.register_object("SeparableCritic", SeparableCritic)
critic_provider.register_object("ScaledDotProduct", ScaledDotProduct)
critic_provider.register_object("CosineSimilarity", CosineSimilarity)


__all__ = [
    "ABaseCritic",
    "SeparableCritic",
    "ScaledDotProduct",
    "CosineSimilarity",
    "critic_provider",
]
