import logging
import nibabel as nib
import numpy as np
import os
import pandas as pd
from typing import List, Optional

from catalyst.data.sampler import BalanceClassSampler
from catalyst.data.loader import BatchPrefetchLoaderWrapper

import torch
from torch.utils.data import DataLoader

import torchio as tio
from torchio import Subject, ScalarImage, SubjectsDataset

from fusion.dataset.abasedataset import ABaseDataset, SetId
from fusion.dataset.utils import seed_worker
from fusion.dataset.oasis.transforms import VolumetricRandomCrop
from fusion.task.misc import TaskId
from fusion.dataset.ukbiobank.transforms import CustomRandomBlur


class UKBioBank(ABaseDataset):
    label_2_name = {
        0: "F:45-52",
        1: "M:45-52",
        2: "F:53-59",
        3: "M:53-59",
        4: "F:60-66",
        5: "M:60-66",
        6: "F:67-73",
        7: "M:67-73",
        8: "F:74-80",
        9: "M:74-80",
    }

    def __init__(
        self,
        dataset_dir: str,
        fold: int = 0,
        num_folds: int = 9,
        sources: List[int] = [0],
        target_sources: List[int] = None,
        input_size: int = 128,
        batch_size: int = 2,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        seed: int = 343,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        num_prefetches: Optional[int] = None,
        use_balanced_sampler: bool = False,
        task_id: TaskId = TaskId.PRETRAINING,
        test_mode: bool = False,
        max_std_blur: float = 5.0,
        skip_sources: List[int] = [2],
        p_blur: float = 1.0,
        mask: str = '/data/users1/xinhui/cpac_dl/out/template/avg152T1_gray_bin.nii.gz',
    ) -> None:
        """_summary_

        Args:
            dataset_dir (str): Path to dataset.
            sources (List[int], optional): List of sources used in the model to support multi-view paradigm. Defaults to [0].
            input_size (int, optional): The input size of the image or volume. Expected to be square. Defaults to 128.
            batch_size (int, optional): The number of samples in the batch. Defaults to 2.
            shuffle (bool, optional): Set True to shuffle the dataset. Defaults to False.
            drop_last (bool, optional): Set True to Drop last batch that smaller than batch size. Defaults to False.
            num_workers (int, optional): The number of subprocesses to use for data loading. Defaults to 0.
            seed (int, optional): The initilization seed for randomness. Defaults to 343.
            prefetch_factor (int, optional): The number of samples has to be preloaded by each each worker. Defaults to 2.
            pin_memory (bool, optional): Set True to copy data to GPU. Defaults to False.
            persistent_workers (bool, optional): Set True to keep dataloading subprocesses alive. Defaults to False.
            num_prefetches (Optional[int], optional): The number of samples to be prefetched into the memory. This option uses Catalyst's BatchPrefetchLoaderWrapper. Defaults to None.
            use_balanced_sampler (bool, optional): Set True to use balanced sampler. This option uses Catalyst's BalanceClassSampler. Defaults to False.
            task_id (TaskId, optional): Defines the strategy of the dataset based on task type. Defaults to TaskId.PRETRAINING.
        """
        super().__init__(
            dataset_dir + f"/fold_{fold}/",
            sources=sources,
            fold=fold,
            num_folds=num_folds,
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
        self._input_size = input_size
        self._use_balanced_sampler = use_balanced_sampler
        self._task_id = task_id
        self._test_mode = test_mode
        self._max_std_blur = max_std_blur
        self._mask = mask
        self._mask_image = None
        print ("MAX_STD_BLUR:", self._max_std_blur)
        self._skip_sources = skip_sources
        self._p_blur = p_blur
        if target_sources is None:
            self._target_sources = self._sources
        else:
            assert len(target_sources) == len(self._sources)
            self._target_sources = target_sources
        self._target_to_source = {}
        for idx, t in enumerate(self._target_sources):
            self._target_to_source[f'{t}_{idx}'] = self._sources[idx]
        print ('Target-2-Source:', self._target_to_source)

    """
    Merged:
    /data/users2/afedorov/trends/fusion/data/UKBioBank
    ----
    /data/users1/xinhui/cpac_dl/script/SMLvsDL/reprex/SampleSplitfMRIPrep2mm/
    te_1600_rep_0.csv  tr_1600_rep_0.csv  va_1600_rep_0.csv
    /data/users1/xinhui/cpac_dl/script/SMLvsDL/reprex/SampleSplitDefault/
    te_1600_rep_0.csv  tr_1600_rep_0.csv  va_1600_rep_0.csv
    /data/users1/xinhui/cpac_dl/script/SMLvsDL/reprex/SampleSplitUKB2mm/
    te_1600_rep_0.csv  tr_1600_rep_0.csv  va_1600_rep_0.csv
    """

    def load(self):
        if os.path.exists(self._mask):
            self._mask_image = nib.load(self._mask)
        else:
            assert False
        for set_id in [SetId.TRAIN, SetId.VALID, SetId.INFER]:
            subjects_list, labels = self._prepare_subject_list(set_id)
            transforms = self._prepare_transforms(set_id)
            dataset = SubjectsDataset(subjects_list, transform=transforms)
            self._set_dataloader(dataset, set_id, labels)

    def _prepare_subject_list(self, set_id: SetId):
        csv_file = os.path.join(self._dataset_dir, f'{set_id}.csv')
        df = pd.read_csv(csv_file)
        subjects_list = self._prepare_list_of_torchio_subjects(df)
        labels = df['label'].values
        if set_id == SetId.TRAIN:
            self._num_classes = len(np.unique(labels))
        return (subjects_list, labels)

    def _prepare_list_of_torchio_subjects(self, df: pd.DataFrame):
        list_of_subjects = []
        print ('Sources:', self._sources)
        indexes = df.index
        if self._test_mode:
            indexes = indexes[:2*self._batch_size]
        for i in indexes:
            subject_dict = {}
            for target, source in self._target_to_source.items():
                target = int(target.split('_')[0])
                filename = df[f"source_{target}"].iloc[i]
                subject_dict[f"source_{source}"] = ScalarImage(filename)
                subject_dict[f"pipeline_{source}"] = target
            # labels should start from 0
            subject_dict["label"] = df.at[i, "label"] - 1
            subject_dict["sex"] = df.at[i, "sex"]
            subject_dict["age"] = df.at[i, "age"]
            subject = Subject(subject_dict)
            list_of_subjects.append(subject)
        print ('Number of subjects:', len(list_of_subjects))
        return list_of_subjects

    def _set_dataloader(
        self,
        dataset: SubjectsDataset,
        set_id: SetId,
        labels: List[int]
    ):
        sampler = None
        if self._use_balanced_sampler:
            sampler = BalanceClassSampler(
                labels,
                mode="downsampling",
            )
        logging.info(
            f"{set_id}:\n"
            f"drop_last={self._drop_last}\n"
            f"shuffle={self._shuffle}\n"
            f"batch_size={self._batch_size}\n"
            f"sampler={sampler}\n"
        )
        if sampler is not None:
            data_loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                drop_last=self._drop_last,
                num_workers=self._num_workers,
                worker_init_fn=seed_worker,
                prefetch_factor=self._prefetch_factor,
                persistent_workers=self._persistent_workers,
                pin_memory=self._pin_memory,
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                drop_last=self._drop_last,
                shuffle=self._shuffle,
                num_workers=self._num_workers,
                worker_init_fn=seed_worker,
                prefetch_factor=self._prefetch_factor,
                persistent_workers=self._persistent_workers,
                pin_memory=self._pin_memory,
            )
        if torch.cuda.is_available() and self._num_prefetches is not None:
            data_loader = BatchPrefetchLoaderWrapper(
                data_loader, num_prefetches=self._num_prefetches
            )
        self._data_loaders[set_id] = data_loader

    def _prepare_transforms(self, set_id: SetId):
        transform = []
        rescale = tio.RescaleIntensity(percentiles=(0, 100))
        transform.append(rescale)
        pad = tio.Pad((18, 19, 9, 10, 18, 19), padding_mode=0)
        transform.append(pad)
        if (self._max_std_blur > 0) and (set_id == SetId.TRAIN):
            blur = CustomRandomBlur(
                std=(0, self._max_std_blur),
                skip_sources=self._skip_sources,
                p=self._p_blur,
            )
            transform.append(blur)
        transform = tio.transforms.Compose(transform)
        return transform
