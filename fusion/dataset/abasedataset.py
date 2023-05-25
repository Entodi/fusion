from .misc import SetId

import abc
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader


class ABaseDataset(abc.ABC):
    def __init__(
        self,
        dataset_dir: str,
        fold: int = 0,
        num_folds: int = 5,
        sources: List[int] = [0],
        batch_size: int = 2,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        seed: int = 343,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        num_prefetches: Optional[int] = None,
    ):
        self._dataset_dir = dataset_dir
        self._fold = fold
        self._num_folds = num_folds
        self._sources = sources
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._num_workers = num_workers
        self._seed = seed
        self._prefetch_factor = prefetch_factor
        self._pin_memory = pin_memory
        self._persistent_workers = persistent_workers
        self._num_prefetches = num_prefetches
        torch.manual_seed(self._seed)
        self._num_classes: Optional[int] = None
        self._data_loaders: Dict[SetId, DataLoader] = {}

    @abc.abstractmethod
    def load(self):
        """Loads the dataset"""
        pass

    def get_all_loaders(self) -> Dict[SetId, DataLoader]:
        """Returns dictionary with data loaders"""
        return self._data_loaders

    def get_cv_loaders(self) -> Dict[SetId, DataLoader]:
        """Returns dictionary with cross-validation loaders"""
        return {
            set_id: self._data_loaders[set_id] for set_id in [SetId.TRAIN, SetId.VALID]
        }

    def get_loader(self, set_id: SetId) -> DataLoader:
        """Returns loader with specific set

        Args:
            set_id (SetID): "\'TRAIN\', \'VALID\', \'TEST\'"
        """
        return self._data_loaders[set_id]

    @property
    def num_classes(self) -> Optional[int]:
        """Number of classes"""
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value: int):
        """Number of classes"""
        self._num_classes = value
