import logging
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from typing import List
import os
import numpy as np

from catalyst.data.sampler import BalanceClassSampler
from catalyst.data.loader import BatchPrefetchLoaderWrapper

import torch
from torch.utils.data import DataLoader

import torchio as tio
from torchio import Subject, ScalarImage, SubjectsDataset

from fusion.dataset.utils import seed_worker
from fusion.dataset.abasedataset import SetId
from fusion.dataset.ukbiobank.transforms import CustomRandomBlur
from . import UKBioBankMP


class UKBioBankMPRAM(UKBioBankMP):

    def _prepare_subject_list(self, set_id: SetId):
        csv_file = os.path.join(self._dataset_dir, f'{set_id}.csv')
        df = pd.read_csv(csv_file)
        subjects_list = self._prepare_list_of_torchio_subjects(set_id, df)
        labels = df['label'].values
        if set_id == SetId.TRAIN:
            self._num_classes = len(np.unique(labels))
        return (subjects_list, labels)

    def _prepare_list_of_torchio_subjects(self, set_id: SetId, df: pd.DataFrame):
        rescale = tio.RescaleIntensity(percentiles=(0, 100))
        pad = tio.Pad((18, 19, 9, 10, 18, 19), padding_mode=0)
        list_of_subjects = []
        print ('Sources:', self._sources)
        indexes = df.index
        if self._test_mode:
            indexes = indexes[:2*self._batch_size]
        for i in tqdm(indexes):
            label = df.at[i, "label"] - 1
            sex = df.at[i, "sex"]
            age = df.at[i, "age"]
            for target, source in self._target_to_source.items():
                target = target.split('_')[0]
                filename = df[f"source_{target}"].iloc[i]
                image = nib.load(filename)
                data = torch.Tensor(image.get_fdata()).unsqueeze(0)
                affine = image.affine
                subject_dict = {}
                scalar_image = ScalarImage(
                    tensor=data, affine=affine)
                scalar_image = rescale(scalar_image)
                scalar_image = pad(scalar_image)
                if set_id != SetId.TRAIN:
                    scalar_image.data = scalar_image.data.cuda()
                subject_dict["source_0"] = scalar_image
                subject_dict["label"] = label
                subject_dict["sex"] = sex
                subject_dict["age"] = age
                subject_dict["pipeline_0"] = source
                subject = Subject(subject_dict)
                list_of_subjects.append(subject)
        print ('Number of subjects:', len(list_of_subjects))
        return list_of_subjects

    def _prepare_transforms(self, set_id: SetId):
        transform = []
        if (self._max_std_blur > 0) and (set_id == SetId.TRAIN):
            blur = CustomRandomBlur(
                std=(0, self._max_std_blur),
                skip_sources=self._skip_sources,
                p=self._p_blur,
            )
            transform.append(blur)
        if len(transform) > 0:
            transform = tio.transforms.Compose(transform)
        else:
            transform = None
        return transform

    def _set_dataloader(
        self,
        dataset: SubjectsDataset,
        set_id: SetId,
        labels: List[int]
    ):
        sampler = None
        pin_memory = self._pin_memory if set_id == SetId.TRAIN else False
        num_workers = self._num_workers if set_id == SetId.TRAIN else 0
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
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                prefetch_factor=self._prefetch_factor,
                persistent_workers=self._persistent_workers,
                pin_memory=pin_memory,
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                drop_last=self._drop_last,
                shuffle=self._shuffle,
                num_workers=num_workers,
                worker_init_fn=seed_worker,
                prefetch_factor=self._prefetch_factor,
                persistent_workers=self._persistent_workers,
                pin_memory=pin_memory,
            )
        if set_id == SetId.TRAIN and torch.cuda.is_available() and self._num_prefetches is not None:
            data_loader = BatchPrefetchLoaderWrapper(
                data_loader, num_prefetches=self._num_prefetches
            )
        self._data_loaders[set_id] = data_loader
