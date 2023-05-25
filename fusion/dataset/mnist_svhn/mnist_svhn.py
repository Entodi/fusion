from catalyst.data.loader import BatchPrefetchLoaderWrapper
import os
from typing import Any, Dict, List, Union, Optional

from sklearn.model_selection import StratifiedKFold
import torch
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchnet.dataset import TensorDataset, ResampleDataset

from fusion.dataset.utils import seed_worker
from fusion.dataset.abasedataset import ABaseDataset
from fusion.dataset.misc import SetId
from fusion.dataset.mnist_svhn.transforms import SVHNTransform, MNISTTransform


class MnistSvhn(ABaseDataset):
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
        """
        Initialization of Class MnistSvhn dataset

        Args:
            dataset_dir: path to dataset
            fold: number of fold for validation
            num_folds: counts of folds
            views: number of views
            batch_size: how many samples per batch to load
            shuffle: set to True to have the data reshuffled at every epoch
            drop_last: set to True to drop the last incomplete batch
            num_workers: how many subprocesses to use for data loading
            seed: number of seed

        Returns:
            Dataset MnistSvhn

        """
        super().__init__(
            dataset_dir + f"MnistSvhn/{fold}/",
            fold=fold,
            num_folds=num_folds,
            sources=sources,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            seed=seed,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_prefetches=num_prefetches,
        )
        self._indexes: Dict[str, Dict[str, Any]] = {}
        self._mnist_dataset_dir = self._dataset_dir + "/MNIST/"
        self._svhn_dataset_dir = self._dataset_dir + "/SVHN/"
        # TODO: this is placeholder to match with scripts on ukbiobank
        self._target_sources = self._sources
        self._target_to_source = {}
        for idx, t in enumerate(self._target_sources):
            self._target_to_source[f'{t}_{idx}'] = self._sources[idx]

    def load(self):
        """
        Method to load dataset
        """
        if not os.path.exists(self._dataset_dir):
            os.makedirs(self._dataset_dir)
            self._download_dataset()
        self._num_classes = 10
        # Don't touch it, otherwise lazy evaluation and lambda functions will make you cry
        samplers = {
            "mnist": {
                SetId.TRAIN: lambda d, i: self._indexes[SetId.TRAIN]["mnist"][i],
                SetId.VALID: lambda d, i: self._indexes[SetId.VALID]["mnist"][i],
                SetId.TEST: lambda d, i: self._indexes[SetId.TEST]["mnist"][i],
            },
            "svhn": {
                SetId.TRAIN: lambda d, i: self._indexes[SetId.TRAIN]["svhn"][i],
                SetId.VALID: lambda d, i: self._indexes[SetId.VALID]["svhn"][i],
                SetId.TEST: lambda d, i: self._indexes[SetId.TEST]["svhn"][i],
            },
        }

        for set_id in [SetId.TRAIN, SetId.VALID, SetId.TEST]:
            dataset = None
            sampler_mnist = samplers["mnist"][set_id]
            sampler_svhn = samplers["svhn"][set_id]
            if len(self._sources) == 2:
                dataset_mnist, indexes_mnist = self._load_subset_dataset(
                    set_id, "mnist"
                )
                dataset_svhn, indexes_svhn = self._load_subset_dataset(set_id, "svhn")
                self._indexes[set_id] = {}
                self._indexes[set_id]["mnist"] = indexes_mnist
                self._indexes[set_id]["svhn"] = indexes_svhn
                dataset = TensorDataset(
                    [
                        ResampleDataset(
                            dataset_mnist.dataset,
                            sampler_mnist,
                            size=len(self._indexes[set_id]["mnist"]),
                        ),
                        ResampleDataset(
                            dataset_svhn.dataset,
                            sampler_svhn,
                            size=len(self._indexes[set_id]["svhn"]),
                        ),
                    ]
                )
                # collate_fn or tensor dataset with transforms
            else:
                if self._sources[0] == 0:
                    dataset_mnist, indexes_mnist = self._load_subset_dataset(
                        set_id, "mnist"
                    )
                    self._indexes[set_id] = {}
                    self._indexes[set_id]["mnist"] = indexes_mnist
                    dataset = TensorDataset(
                        [
                            ResampleDataset(
                                dataset_mnist.dataset,
                                sampler_mnist,
                                size=len(indexes_mnist),
                            ),
                        ]
                    )
                elif self._sources[0] == 1:
                    self._indexes[set_id] = {}
                    dataset_svhn, indexes_svhn = self._load_subset_dataset(
                        set_id, "svhn"
                    )
                    self._indexes[set_id]["svhn"] = indexes_svhn
                    dataset = TensorDataset(
                        [
                            ResampleDataset(
                                dataset_svhn.dataset,
                                sampler_svhn,
                                size=len(indexes_svhn),
                            )
                        ]
                    )
            self._set_dataloader(dataset, set_id)

    def _load_subset_dataset(self, set_id: SetId, dataset_name: str):
        # define filename for pair indexes
        if set_id != SetId.TEST:
            filename = f"{set_id}-ms-{dataset_name}-idx-{self._fold}.pt"
        else:
            filename = f"{set_id}-ms-{dataset_name}-idx.pt"
        # load paired indexes
        indexes = torch.load(os.path.join(self._dataset_dir, filename))
        # load dataset
        if dataset_name == "mnist":
            # validation uses training set
            train = True if set_id != SetId.TEST else False
            tx = MNISTTransform()
            dataset = torchvision.datasets.MNIST(
                self._mnist_dataset_dir, train=train, download=False, transform=tx
            )
        elif dataset_name == "svhn":
            # validation uses training set
            split = SetId.TRAIN if set_id != SetId.TEST else SetId.TEST
            tx = SVHNTransform()
            dataset = torchvision.datasets.SVHN(
                self._svhn_dataset_dir, split=split, download=False, transform=tx
            )
        else:
            raise NotImplementedError
        # select fold
        if set_id != SetId.TEST:
            cv_indexes = torch.load(
                os.path.join(
                    self._dataset_dir,
                    f"{set_id}-ms-{dataset_name}-cv-idx-{self._fold}.pt",
                )
            )
            dataset.data = dataset.data[cv_indexes]
            if dataset_name == "mnist":
                dataset.targets = dataset.targets[cv_indexes]
            elif dataset_name == "svhn":
                dataset.labels = dataset.labels[cv_indexes]
            else:
                raise NotImplementedError
        dataset = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
            worker_init_fn=seed_worker,
        )
        return dataset, indexes

    def _set_dataloader(self, dataset: Dataset, set_id: SetId):
        drop_last = True if set_id == SetId.TRAIN else self._drop_last
        shuffle = True if set_id == SetId.TRAIN else self._shuffle
        data_loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self._num_workers,
            worker_init_fn=seed_worker,
            prefetch_factor=self._prefetch_factor,
            persistent_workers=self._persistent_workers,
            pin_memory=self._pin_memory,
        )
        set_id = SetId.INFER if set_id == SetId.TEST else set_id
        if torch.cuda.is_available():
            data_loader = BatchPrefetchLoaderWrapper(
                data_loader, num_prefetches=self._num_prefetches
            )
        self._data_loaders[set_id] = data_loader

    def _set_num_classes(self, targets: Tensor):
        self._num_classes = len(torch.unique(targets))

    def _prepare_fold(
        self,
        dataset: Union[torchvision.datasets.MNIST, torchvision.datasets.SVHN],
        dataset_name: str,
    ):
        kf = StratifiedKFold(
            n_splits=self._num_folds, shuffle=self._shuffle, random_state=self._seed
        )
        if dataset_name == "MNIST":
            X, y = dataset.data, dataset.targets
        else:
            X, y = dataset.data, dataset.labels
        kf_g = kf.split(X, y)
        for _ in range(1, self._fold):
            next(kf_g)
        train_index, valid_index = next(kf.split(X, y))
        return train_index, valid_index

    @staticmethod
    def _rand_match_on_idx(l1, idx1, l2, idx2, max_d: int = 10000, dm: int = 10):
        """
        Args:
            l*: sorted labels
            idx*: indices of sorted labels in original list
        """
        _idx1, _idx2 = [], []
        for l in l1.unique():  # assuming both have same idxs
            l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
            n = min(l_idx1.size(0), l_idx2.size(0), max_d)
            l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
            for _ in range(dm):
                _idx1.append(l_idx1[torch.randperm(n)])
                _idx2.append(l_idx2[torch.randperm(n)])
        return torch.cat(_idx1), torch.cat(_idx2)

    def _download_dataset(self):
        max_d = 10000  # maximum number of datapoints per class
        dm = 30  # data multiplier: random permutations to match

        # get the individual datasets
        tx = torchvision.transforms.ToTensor()
        if os.path.exists(self._mnist_dataset_dir):
            download = False
        else:
            download = True
            os.mkdir(self._mnist_dataset_dir)
        # load mnist
        train_mnist = torchvision.datasets.MNIST(
            self._mnist_dataset_dir, train=True, download=download, transform=tx
        )
        test_mnist = torchvision.datasets.MNIST(
            self._mnist_dataset_dir, train=False, download=download, transform=tx
        )

        if os.path.exists(self._svhn_dataset_dir):
            download = False
        else:
            download = True
            os.mkdir(self._svhn_dataset_dir)
        # load svhn
        train_svhn = torchvision.datasets.SVHN(
            self._svhn_dataset_dir, split=SetId.TRAIN, download=download, transform=tx
        )
        test_svhn = torchvision.datasets.SVHN(
            self._svhn_dataset_dir, split=SetId.TEST, download=download, transform=tx
        )

        # svhn labels need extra work
        train_svhn.labels = (
            torch.LongTensor(train_svhn.labels.squeeze().astype(int)) % 10
        )
        test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze().astype(int)) % 10

        # split on cross-validation folds
        mnist_train_idxs, mnist_valid_idxs = self._prepare_fold(train_mnist, "MNIST")
        svhn_train_idxs, svhn_valid_idxs = self._prepare_fold(train_svhn, "SVHN")

        # save and pair training set
        mnist_l, mnist_li = train_mnist.targets[mnist_train_idxs].sort()
        svhn_l, svhn_li = train_svhn.labels[svhn_train_idxs].sort()
        idx1, idx2 = self._rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm
        )
        torch.save(
            idx1, os.path.join(self._dataset_dir, f"train-ms-mnist-idx-{self._fold}.pt")
        )
        torch.save(
            idx2, os.path.join(self._dataset_dir, f"train-ms-svhn-idx-{self._fold}.pt")
        )
        torch.save(
            mnist_train_idxs,
            os.path.join(self._dataset_dir, f"train-ms-mnist-cv-idx-{self._fold}.pt"),
        )
        torch.save(
            svhn_train_idxs,
            os.path.join(self._dataset_dir, f"train-ms-svhn-cv-idx-{self._fold}.pt"),
        )

        # save and pair validation set
        mnist_l, mnist_li = train_mnist.targets[mnist_valid_idxs].sort()
        svhn_l, svhn_li = train_svhn.labels[svhn_valid_idxs].sort()
        idx1, idx2 = self._rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm
        )
        torch.save(
            idx1, os.path.join(self._dataset_dir, f"valid-ms-mnist-idx-{self._fold}.pt")
        )
        torch.save(
            idx2, os.path.join(self._dataset_dir, f"valid-ms-svhn-idx-{self._fold}.pt")
        )
        torch.save(
            mnist_valid_idxs,
            os.path.join(self._dataset_dir, f"valid-ms-mnist-cv-idx-{self._fold}.pt"),
        )
        torch.save(
            svhn_valid_idxs,
            os.path.join(self._dataset_dir, f"valid-ms-svhn-cv-idx-{self._fold}.pt"),
        )

        # save and pair test set
        mnist_l, mnist_li = test_mnist.targets.sort()
        svhn_l, svhn_li = test_svhn.labels.sort()
        idx1, idx2 = self._rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm
        )
        torch.save(idx1, os.path.join(self._dataset_dir, "test-ms-mnist-idx.pt"))
        torch.save(idx2, os.path.join(self._dataset_dir, "test-ms-svhn-idx.pt"))
