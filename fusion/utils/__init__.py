from .object_provider import ObjectProvider
from collections import namedtuple


Setting = namedtuple("Setting", ["class_type", "args"])


__all__ = ["ObjectProvider", "Setting"]
