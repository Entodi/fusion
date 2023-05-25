from .catalyst import CatalystRunner, MnistSvhnRunner, OasisRunner
from fusion.utils import ObjectProvider


runner_provider = ObjectProvider()
runner_provider.register_object("CatalystRunner", CatalystRunner)
runner_provider.register_object("MnistSvhnRunner", MnistSvhnRunner)
runner_provider.register_object("OasisRunner", OasisRunner)