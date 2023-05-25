from catalyst.contrib.optimizers.radam import RAdam
from torch.optim import SGD, Adam, RMSprop
from fusion.utils import ObjectProvider


optimizer_provider = ObjectProvider()
optimizer_provider.register_object("RAdam", RAdam)
optimizer_provider.register_object("Adam", Adam)
optimizer_provider.register_object("SGD", SGD)
optimizer_provider.register_object("RMSprop", RMSprop)
