from . import UKBioBank

import os
import pandas as pd
import numpy as np
from typing import List, Optional

from fusion.dataset.abasedataset import SetId
from fusion.task.misc import TaskId

import torch
from torchio import Subject, ScalarImage, SubjectsDataset


class UKBioBankAGP(UKBioBank):
    # Any Group Pair UKBioBank
    def __init__(
        self,
        dataset_dir: str,
        fold: int = 0,
        num_folds: int = 9,
        sources: List[int] = [0, 1, 2],
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
        max_datapoints_per_class: int = 10000,
        num_matching_random_permutation: int = 30
    ) -> None:
        super().__init__(
            dataset_dir,
            fold=fold,
            num_folds=num_folds,
            sources=sources,
            input_size=input_size,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            seed=seed,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            num_prefetches=num_prefetches,
            use_balanced_sampler=use_balanced_sampler,
            task_id=task_id,
            test_mode=test_mode,
        )
        assert len(self._sources) == 2, 'Only 2 sources are supported'
        self._max_datapoints_per_class = max_datapoints_per_class
        self._num_matching_random_permutation = num_matching_random_permutation

    def _prepare_subject_list(self, set_id: SetId):
        csv_file = os.path.join(self._dataset_dir, f'{set_id}.csv')
        df = pd.read_csv(csv_file)
        l = torch.LongTensor(df['label'].values)
        idx = torch.IntTensor(df.index)
        if set_id in [SetId.VALID, SetId.INFER]:
            dm = 1
        else:
            dm = self._num_matching_random_permutation
        new_idx1, new_idx2 = self._rand_match_on_idx(
            l, idx, l, idx,
            max_d=self._max_datapoints_per_class,
            dm=dm,
        )
        print (new_idx1[0], new_idx2[0], l[new_idx1[0]], l[new_idx2[0]])
        new_df1 = df.iloc[new_idx1].reset_index(drop=True)
        new_df2 = df.iloc[new_idx2].reset_index(drop=True)
        subjects_list = self._prepare_list_of_torchio_subjects(
            [new_df1, new_df2])
        labels = df['label'].values
        if set_id == SetId.TRAIN:
            self._num_classes = len(np.unique(labels))
        return (subjects_list, labels)

    def _prepare_list_of_torchio_subjects(self, dfs: List[pd.DataFrame]):
        list_of_subjects = []
        print ('Sources:', self._sources)
        indexes = dfs[0].index
        if self._test_mode:
            indexes = indexes[:2*self._batch_size]
        for i in indexes:
            subject_dict = {}
            for sj, source_id in enumerate(self._sources):
                filename = dfs[sj][f"source_{source_id}"].iloc[i]
                subject_dict[f"source_{source_id}"] = ScalarImage(filename)
            # labels should start from 0
            subject_dict["label"] = dfs[0].at[i, "label"] - 1
            subject = Subject(subject_dict)
            list_of_subjects.append(subject)
        print ('Number of subjects:', len(list_of_subjects))
        return list_of_subjects

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
