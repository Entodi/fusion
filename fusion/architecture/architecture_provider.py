from .dcgan import DcganEncoder, DcganDecoder, DcganAutoEncoder

from .alexnet import AlexNetEncoder, AneesAlexNet
from fusion.utils import ObjectProvider


architecture_provider = ObjectProvider()
architecture_provider.register_object("DcganEncoder", DcganEncoder)
architecture_provider.register_object("DcganDecoder", DcganDecoder)
architecture_provider.register_object("DcganAutoEncoder", DcganAutoEncoder)
architecture_provider.register_object("AlexNetEncoder", AlexNetEncoder)
architecture_provider.register_object("AneesAlexNet", AneesAlexNet)