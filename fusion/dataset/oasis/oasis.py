import logging
import nibabel as nib
import numpy as np
import os
import pandas as pd
from typing import List, Optional, Dict

from catalyst.data.sampler import BalanceClassSampler
from catalyst.data.loader import BatchPrefetchLoaderWrapper

import torch
from torch import Tensor
from torch.utils.data import DataLoader

import torchio as tio
from torchio import Subject, ScalarImage, SubjectsDataset

from fusion.dataset.abasedataset import ABaseDataset, SetId
from fusion.dataset.utils import seed_worker
from fusion.task.misc import TaskId

from .transforms import MNIMaskTransform
from .transforms import VolumetricRandomCrop


class Oasis(ABaseDataset):
    def __init__(
        self,
        dataset_dir: str,
        fold: int = 0,
        num_folds: int = 5,
        sources: List[int] = [0],
        input_size: int = 64,
        batch_size: int = 2,
        shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        seed: int = 343,
        prefetch_factor: int = 2,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        num_prefetches: Optional[int] = None,
        is_only_one_pair_per_subject: bool = False,
        use_balanced_sampler: bool = False,
        use_separate_augmentation: bool = False,
        only_labeled: bool = False,
        task_id: TaskId = TaskId.PRETRAINING,
        unlabelled_as_class: bool = False,
        mask: str = './data/MNI152_T1_3mm_brain_mask_dil_cubic192.nii.gz',
        transforms: Dict[str, bool] = None,
        target_sources: List[int] = None,
    ):
        """
        Initialization of Class Oasis dataset

        Args:
            dataset_dir: path to dataset
            fold: number of fold for validation
            num_folds: counts of folds
            source_ids: number of source_ids
            batch_size: how many samples per batch to load
            shuffle: set to True to have the data reshuffled at every epoch
            drop_last: set to True to drop the last incomplete batch
            num_workers: how many subprocesses to use for data loading
            seed: number of seed
            is_only_one_pair_per_subject: set to True to use only one pair (given pandas algorithm uses first entry in the dataframe) of images per subjects
            use_balanced_sampler: set to True to use balanced data sampler
            use_separate_augmentation: set to True to have different augmentation for modalities
            only_labeled: set to True to remove unlabeled ("-1") samples
            task_id: the task id, needed for transforms

        Returns:
            Dataset Oasis

        """
        super().__init__(
            dataset_dir + f"/fold_{fold}/",
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
        assert (
            not ((shuffle is True) and (use_balanced_sampler is True))
        ), "Sampler and Shuffle do not go together for dataloader in PyTorch"
        self._input_size = input_size
        self._is_only_one_pair_per_subject = is_only_one_pair_per_subject
        self._use_balanced_sampler = use_balanced_sampler
        self._use_separate_augmentation = use_separate_augmentation
        self._unlabelled_as_class = unlabelled_as_class
        self._only_labeled = only_labeled
        self._task_id = task_id
        self._mask = mask
        self._mask_image = None
        self._transforms = transforms
        self._landmarks = {}
        if target_sources is None:
            self._target_sources = self._sources
        else:
            assert len(target_sources) == len(self._sources)
            self._target_sources = target_sources
        self._target_to_source = {}
        for idx, t in enumerate(self._target_sources):
            self._target_to_source[f'{t}_{idx}'] = self._sources[idx]
        logging.info(
            f"is_only_one_pair_per_subject: {is_only_one_pair_per_subject}\n"
            f"use_balanced_sampler: {use_balanced_sampler}\n"
            f"only_labeled: {only_labeled}\n"
        )

    def load(self, only_data=False):
        if os.path.exists(self._mask):
            self._mask_image = nib.load(self._mask)
            #print (self._mask_image.header)
            #print (nib.aff2axcodes(self._mask_image.affine))
            # reorient to RAS+
            #self._mask_image = nib.as_closest_canonical(self._mask_image)
            #print (nib.aff2axcodes(self._mask_image.affine))
            #print (self._mask_image.header)
        else:
            assert False
        data = {}
        for set_id in [SetId.TRAIN, SetId.VALID, SetId.INFER]:
            list_of_subjects, labels, df = self._prepare_subject_list(
                set_id, only_data=only_data)
            if not only_data:
                transforms = self._prepare_transforms(
                    set_id, self._use_separate_augmentation
                )
                dataset = SubjectsDataset(list_of_subjects, transform=transforms)
                self._set_dataloader(dataset, set_id, labels)
            data[set_id] = df
        return data

    def _set_dataloader(
        self, dataset: SubjectsDataset, set_id: SetId, labels: List[int]
    ):
        sampler = None
        drop_last = self._drop_last
        shuffle = self._shuffle
        if set_id != SetId.INFER:
            sampler = (
                BalanceClassSampler(
                    labels,
                    mode="upsampling",
                )
                if self._use_balanced_sampler
                else None
            )
        logging.info(
            f"{set_id}:\n"
            f"drop_last={drop_last}\n"
            f"shuffle={shuffle}\n"
            f"batch_size={self._batch_size}\n"
            f"sampler={sampler}\n"
        )
        if sampler is not None:
            data_loader = DataLoader(
                dataset,
                batch_size=self._batch_size,
                drop_last=drop_last,
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
                drop_last=drop_last,
                shuffle=shuffle,
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

    def _prepare_subject_list(self, set_id: SetId, only_data=False):
        df = self._load_csv(self._dataset_dir, set_id)
        logging.info(f"Loaded dataset with {df.shape[0]} inputs.")
        if (set_id == SetId.INFER) or (self._is_only_one_pair_per_subject):
            df = self._drop_duplicate_pairs(df)
        if self._only_labeled:
            df = self._keep_only_labeled(df)
        list_of_subjects = None
        if not only_data:
            list_of_subjects = self._prepare_list_of_torchio_subjects(df, self._sources)
        labels = df["target"].values
        if set_id == SetId.TRAIN:
            self._set_num_classes(labels)
        return (list_of_subjects, labels, df)

    def _train_histogram_standartization(self):
        train_dataset_csv = os.path.join(
            self._dataset_dir, 'train.csv'
        )
        train_dataset = pd.read_csv(train_dataset_csv)
        for source in self._sources:
            column = f"filename_{source + 1}"
            df = train_dataset
            df = df.drop_duplicates(subset=column)
            df = df.reset_index(drop=True)
            paths = df[column].values
            source_landmarks_path = os.path.abspath(os.path.join(
                self._dataset_dir, f'landmarks_{source + 1}.npy'
            ))
            if not os.path.exists(source_landmarks_path):
                logging.info("Training Histogram Standardization")
                landmarks = tio.HistogramStandardization.train(
                    paths, output_path=source_landmarks_path,
                )
            else:
                logging.info("Loading Histogram Standardization landmarks")
                landmarks = np.load(source_landmarks_path)
            self._landmarks[f'source_{source}'] = landmarks

    def _prepare_transforms2(
        self, set_id: SetId, use_separate_augmentation: bool = False
    ):
        assert (
            use_separate_augmentation is False
        ), "Separate augmentations have not been implemented"
        self.landmarks = {}
        self._train_histogram_standartization()
        #canonical = tio.transforms.ToCanonical()
        mask = MNIMaskTransform(template=self._mask)
        hist_standard = tio.transforms.HistogramStandardization(
            self._landmarks)
        znorm = tio.transforms.ZNormalization(
            masking_method=tio.transforms.ZNormalization.mean
        )
        pad_size = self._input_size // 8
        pad = tio.transforms.Pad(
            padding=( 
                pad_size, pad_size,
                pad_size, pad_size,
                pad_size, pad_size
            ),
            padding_mode='reflect'
        )
        crop = VolumetricRandomCrop(self._input_size)
        flip = tio.transforms.RandomFlip(axes=(0, 1, 2), p=0.5)
        rescale = tio.RescaleIntensity(percentiles=(0, 100))
        transforms = [mask]
        #transforms.append(hist_standard)
        transforms.append(rescale)
        transforms.append(znorm)
        if set_id == SetId.TRAIN and self._task_id == TaskId.PRETRAINING:
            transforms.append(pad)
            transforms.append(crop)
            transforms.append(flip)
        transforms = tio.transforms.Compose(transforms)
        return transforms

    def _prepare_transforms(
        self, set_id: SetId, use_separate_augmentation: bool = False
    ):
        accepted_transforms = [
            'pad_crop', 'flip', 'rescale', 'histogram', 'z_normalization', 'mask'
        ]
        for key in self._transforms.keys():
            assert key in accepted_transforms
        assert (
            use_separate_augmentation is False
        ), "Separate augmentations have not been implemented"
        transforms = []
        # ----------------------------------------------------------------------------
        if self._transforms['mask']:
            mask = MNIMaskTransform(template=self._mask)
            transforms.append(mask)
        # ----------------------------------------------------------------------------
        if self._transforms['rescale']:
            rescale = tio.RescaleIntensity(percentiles=(0, 100))
            transforms.append(rescale)
        # ----------------------------------------------------------------------------
        if self._transforms['histogram']:
            if set_id == SetId.TRAIN:
                self.landmarks = {}
                self._train_histogram_standartization()
            hist_standard = tio.transforms.HistogramStandardization(
                self._landmarks)
            transforms.append(hist_standard)
        # ----------------------------------------------------------------------------
        if self._transforms['z_normalization']:
            znorm = tio.transforms.ZNormalization(
                masking_method=tio.transforms.ZNormalization.mean
            )
            transforms.append(znorm)
        # ----------------------------------------------------------------------------
        if self._task_id == TaskId.PRETRAINING:
            # ----------------------------------------------------------------------------
            if self._transforms['pad_crop']:
                pad_size = self._input_size // 8
                pad = tio.transforms.Pad(
                    padding=(
                        pad_size, pad_size,
                        pad_size, pad_size,
                        pad_size, pad_size
                    ),
                    padding_mode=0
                )
                crop = VolumetricRandomCrop(self._input_size)
                transforms.append(pad)
                transforms.append(crop)
            # ----------------------------------------------------------------------------
            if self._transforms['flip']:
                flip = tio.transforms.RandomFlip(axes=(0, 1, 2), p=0.5)
                transforms.append(flip)
        # ----------------------------------------------------------------------------
        transforms = tio.transforms.Compose(transforms)
        if len(transforms) == 0:
            transforms = None
        print ("Transforms:", set_id, transforms)
        return transforms

    @staticmethod
    def _drop_duplicate_pairs(df: pd.DataFrame):
        logging.info(f"Shape with multiple pairs per subject {df.shape}")
        df = df.drop_duplicates(subset="subject").reset_index(drop=True)
        logging.info(f"Shape with only one pair per subject {df.shape}")
        return df

    @staticmethod
    def _keep_only_labeled(df: pd.DataFrame):
        logging.info("Cleaning labels with -1")
        df = df[df["target"] != -1].reset_index(drop=True)
        logging.info(f"Shape without label -1 {df.shape}")
        logging.info(df["target"].value_counts())
        return df

    @staticmethod
    def _load_csv(dataset_dir: str, set_id: SetId):
        csv_file = os.path.join(dataset_dir, f"{set_id}.csv")
        df = pd.read_csv(csv_file)
        return df

    def _prepare_list_of_torchio_subjects(self, df: pd.DataFrame, sources: List[int]):
        list_of_subjects = []
        for i in df.index:
            subject_dict = {}
            for source_id in sources:
                filename = df[f"filename_{source_id + 1}"].iloc[i]
                # print (filename)
                subject_dict[f"source_{source_id}"] = ScalarImage(filename)
            label = df.at[i, "target"]
            if self._unlabelled_as_class and label == -1:
                label = 2
            subject_dict["label"] = label
            assert df.at[i, "M/F"] in ['M', 'F']
            subject_dict["gender"] = 0 if df.at[i, "M/F"] == 'M' else 1
            subject_dict["age"] = df.at[i, "ageAtEntry"]
            subject = Subject(subject_dict)
            list_of_subjects.append(subject)
        return list_of_subjects

    def _set_num_classes(self, targets: Tensor):
        self._num_classes = len(np.unique(targets))
        if self._unlabelled_as_class:
            targets[targets == -1] = self._num_classes

    @property
    def mask(self):
        return self._mask_image

    @mask.setter
    def mask(self, new_mask: nib.Nifti1Image):
        self._mask_image = new_mask

    def header(self):
        return self._mask_image.header

    def affine(self):
        return self._mask_image.affine

    def orientation(self):
        return nib.aff2axcodes(self.affine())
