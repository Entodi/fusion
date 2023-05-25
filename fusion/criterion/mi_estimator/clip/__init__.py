from .base_clip import ABaseClip
from .tahn_clip import TahnClip

from fusion.utils import ObjectProvider

clip_provider = ObjectProvider()
clip_provider.register_object("TahnClip", TahnClip)


__all__ = ["ABaseClip", "TahnClip", "clip_provider"]
