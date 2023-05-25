from . import UKBioBank

import pandas as pd
from typing import List, Optional

from torchio import Subject, ScalarImage

from fusion.task.misc import TaskId


class UKBioBankMP(UKBioBank):
    # Multi Pipeline Versions of the UKBioBank
    # Uses all 3 sources by default [0, 1, 2]
    def __init__(
        self,
        dataset_dir: str,
        fold: int = 0,
        num_folds: int = 9,
        sources: List[int] = [0, 1, 2],
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
        p_blur: float = 1.0,
        skip_sources: List[int] = [2],
    ) -> None:
        super().__init__(
            dataset_dir,
            fold=fold,
            num_folds=num_folds,
            sources=sources,
            target_sources=target_sources,
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
            max_std_blur=max_std_blur,
            p_blur=p_blur,
            skip_sources=skip_sources,
        )

    def _prepare_list_of_torchio_subjects(self, df: pd.DataFrame):
        list_of_subjects = []
        print ('Sources:', self._sources)
        indexes = df.index
        if self._test_mode:
            indexes = indexes[:2*self._batch_size]
        for i in indexes:
            label = df.at[i, "label"] - 1
            sex = df.at[i, "sex"]
            age = df.at[i, "age"]
            for target, source in self._target_to_source.items():
                target = target.split('_')[0]
                filename = df[f"source_{target}"].iloc[i]
                subject_dict = {}
                subject_dict[f"source_0"] = ScalarImage(filename)
                subject_dict["label"] = label
                subject_dict["sex"] = sex
                subject_dict["age"] = age
                subject_dict["pipeline_0"] = source
                subject = Subject(subject_dict)
                list_of_subjects.append(subject)
        print ('Number of subjects:', len(list_of_subjects))
        return list_of_subjects
