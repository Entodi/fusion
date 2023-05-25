from .supervised import Supervised
from .ae import AE
from .dim import Dim
from .linear_evaluator import LinearEvaluator
from .encoder_extractor import EncoderExtractor
from .pixl import PIXL
from .saliency import SmoothVanillaBackprop
from fusion.utils import ObjectProvider


model_provider = ObjectProvider()
model_provider.register_object("Supervised", Supervised)
model_provider.register_object("AE", AE)
model_provider.register_object("Dim", Dim)
model_provider.register_object("LinearEvaluator", LinearEvaluator)
model_provider.register_object("EncoderExtractor", EncoderExtractor)
model_provider.register_object("SmoothGrad", SmoothVanillaBackprop)
model_provider.register_object("PIXL", PIXL)
