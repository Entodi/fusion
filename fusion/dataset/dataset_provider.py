from .two_view_mnist import TwoViewMnist
from .mnist_svhn import MnistSvhn
from .oasis import Oasis
from .ukbiobank import UKBioBank, UKBioBankMP, UKBioBankAGP, \
    UKBioBankRAM, UKBioBankMPRAM

from fusion.utils import ObjectProvider


dataset_provider = ObjectProvider()
dataset_provider.register_object("TwoViewMnist", TwoViewMnist)
dataset_provider.register_object("MnistSvhn", MnistSvhn)
dataset_provider.register_object("Oasis", Oasis)
dataset_provider.register_object("UKBioBank", UKBioBank)
dataset_provider.register_object("UKBioBankMP", UKBioBankMP)
dataset_provider.register_object("UKBioBankAGP", UKBioBankAGP)
dataset_provider.register_object("UKBioBankRAM", UKBioBankRAM)
dataset_provider.register_object("UKBioBankMPRAM", UKBioBankMPRAM)
