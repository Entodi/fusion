from fusion.utils import ObjectProvider
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


scheduler_provider = ObjectProvider()
scheduler_provider.register_object("OneCycleLR", OneCycleLR)
scheduler_provider.register_object("CAWR", CosineAnnealingWarmRestarts)
scheduler_provider.register_object("CLR", CyclicLR)
scheduler_provider.register_object("RLRP", ReduceLROnPlateau)
