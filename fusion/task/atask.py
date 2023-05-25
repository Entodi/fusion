import abc
from catalyst.utils.misc import set_global_seed
import copy
import numpy as np
from omegaconf import SCMode
from omegaconf import OmegaConf
from omegaconf import DictConfig
import random
import torch
from typing import Optional, Dict


class ATask(abc.ABC):
    _dataset = None
    _model = None
    _criterion = None
    _optimizer = None
    _scheduler = None
    _runner = None
    _callbacks = None
    _loggers = None

    def __init__(self, config: DictConfig, task_args: DictConfig, seed: int = 343):
        self._config = config
        self._task_args = task_args
        self._seed = seed

    @abc.abstractmethod
    def run(self):
        pass

    def _reset_seed(self):
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed_all(self._seed)
        set_global_seed(self._seed)

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion

    @property
    def runner(self):
        return self._runner

    @runner.setter
    def runner(self, runner):
        self._runner = runner

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        self._scheduler = scheduler

    @property
    def task_args(self):
        return self._task_args

    def _log_hparams2(self, logger):
        def unroll(tree, prefix='', hparams={}):
            print (tree.keys())
            print ('\n')
            for key, params in tree.items():
                print (key, params)
                print ('\n')
                if type(params) is DictConfig:
                    hparams = unroll(
                        params, prefix=f"{prefix}{key}_", hparams=hparams
                    )
                else:
                    hparams[f"{prefix}{key}"] = params
            return hparams

        hparams = unroll(self._config)
        if logger is not None:
            logger.log_hparams(hparams)

    # https://github.com/wandb/client/issues/1233#issuecomment-693205205
    def _log_hparams(self, logger):
        hparams = OmegaConf.to_container(self._config, structured_config_mode=SCMode.DICT_CONFIG, resolve=True)
        print (hparams)
        if logger is not None:
            logger.log_hparams(hparams)


class ATaskBuilder(abc.ABC):
    _task: Optional[ATask] = None

    @abc.abstractmethod
    def create_new_task(self, args):
        pass

    @abc.abstractmethod
    def add_dataset(self, dataset_config):
        pass

    @abc.abstractmethod
    def add_model(self, model_config):
        pass

    @abc.abstractmethod
    def add_criterion(self, criterion_config):
        pass

    @abc.abstractmethod
    def add_runner(self, runner_config):
        pass

    @abc.abstractmethod
    def add_optimizer(self, optimizer_config):
        pass

    @abc.abstractmethod
    def add_scheduler(self, scheduler_config):
        pass

    @property
    def task(self):
        return self._task


class TaskDirector:
    def __init__(self, task_builder: ATaskBuilder, config: DictConfig, seed: int = 343):
        self._builder = task_builder
        self._config = config
        self._seed = seed

    def construct_task(self):
        self._builder.create_new_task(self._config, self._config.task, seed=self._seed)
        self._builder.add_dataset(self._config.dataset)
        self._builder.add_model(self._config.model)
        if 'criterion' in self._config.keys():
            self._builder.add_criterion(self._config.criterion)
        if 'optimizer' in self._config.keys():
            self._builder.add_optimizer(self._config.optimizer)
        if 'scheduler' in self._config.keys():
            self._builder.add_scheduler(self._config.scheduler)
        if 'runner' in self._config.keys():
            self._builder.add_runner(self._config.runner)

    def get_task(self):
        return self._builder.task
