from .base_dim import BaseDim
from .cr_dim import CrDim, CR_MODE
from .xx_dim import XxDim, XX_MODE
from .cc_dim import CcDim, CC_MODE
from .rr_dim import RrDim, RR_MODE
from fusion.utils import ObjectProvider


dim_mode_provider = ObjectProvider()
dim_mode_provider.register_object(RR_MODE, RrDim)
dim_mode_provider.register_object(CR_MODE, CrDim)
dim_mode_provider.register_object(XX_MODE, XxDim)
dim_mode_provider.register_object(CC_MODE, CcDim)


__all__ = [
    "BaseDim",
    "CrDim",
    "XxDim",
    "CcDim",
    "RrDim",
    "CR_MODE",
    "RR_MODE",
    "CC_MODE",
    "XX_MODE",
    "dim_mode_provider",
]
