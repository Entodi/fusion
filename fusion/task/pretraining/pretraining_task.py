from catalyst import dl
from catalyst.callbacks.checkpoint import CheckpointCallback
from catalyst.loggers.wandb import WandbLogger

from omegaconf import DictConfig
from fusion.dataset import dataset_provider
from fusion.dataset.misc import SetId
from fusion.model import model_provider
from fusion.model.supervised import Supervised
from fusion.model.pixl import PIXL
from fusion.criterion import criterion_provider
from fusion.optimizer import optimizer_provider
from fusion.runner import runner_provider
from fusion.scheduler import scheduler_provider
from fusion.task import ATask, ATaskBuilder
import logging

import time


class PretrainingTaskBuilder(ATaskBuilder):
    _task: ATask

    def create_new_task(self, config: DictConfig, task_args: DictConfig, seed: int = 343):
        """
        Method for create new pretraining task

        Args:
            task_args: dictionary with task's parameters from config
        """
        self._task = PretrainingTask(config, task_args.args, seed=seed)

    def add_dataset(self, dataset_config: DictConfig):
        """
        Method for add dataset to pretraining task

        Args:
            dataset_config: dictionary with dataset's parameters from config
        """
        self._task.dataset = dataset_provider.get(
            dataset_config.name, **dataset_config.args
        )
        self._task.dataset.load()

    def add_model(self, model_config: DictConfig):
        """
        Method for add model to pretraining task

        Args:
            model_config: dictionary with model's parameters from config
        """
        if "num_classes" in model_config.args.keys():
            model_config.args["num_classes"] = self._task.dataset._num_classes
        model_args = {**model_config.args}
        model_args.pop("pretrained_checkpoint")
        self._task.model = model_provider.get(model_config.name, **model_args)
        print(self._task.model)

    def add_criterion(self, criterion_config: DictConfig):
        """
        Method for add criterion to pretraining task

        Args:
            criterion_config: dictionary with criterion's parameters from config
        """
        args = {} if criterion_config.args is None else criterion_config.args
        self._task.criterion = criterion_provider.get(criterion_config.name, **args)

    def add_runner(self, runner_config: DictConfig):
        """
        Method for add runner to pretraining task

        Args:
            runner_config: dictionary with runner's parameters from config
        """
        runner_args = {} if runner_config.args is None else runner_config.args
        self._task.runner = runner_provider.get(runner_config.name, **runner_args)

    def add_optimizer(self, optimizer_config: DictConfig):
        """
        Method for add optimizer to pretraining task

        Args:
            optimizer_config: dictionary with optimizer's parameters from config
        """
        args = dict(**optimizer_config.args)
        args["params"] = self._task.model.parameters()
        self._task.optimizer = optimizer_provider.get(optimizer_config.name, **args)

    def add_scheduler(self, scheduler_config: DictConfig):
        """
        Method for add scheduler to pretraining task
        
        Args:
            scheduler_config: dictionary with scheduler's parameters from config
        """
        args = dict(**scheduler_config.args)
        args["optimizer"] = self._task.optimizer
        print(scheduler_config.name)
        if scheduler_config.name == "OneCycleLR":
            args["steps_per_epoch"] = len(self._task.dataset.get_loader(SetId.TRAIN))
            args["epochs"] = self._task.task_args["num_epochs"]
        elif scheduler_config.name in ["CAWR", "CLR", "RLRP"]:
            pass
        else:
            raise NotImplementedError
        self._task.scheduler = scheduler_provider.get(
            scheduler_config.name, **args
        )


class PretrainingTask(ATask):
    def run(self):
        """
        Method launch training of Pretraining Task
        """
        wandb_logger = WandbLogger(
            project=self._task_args["project"],
            name=self._task_args["name"],
            entity=self._task_args["entity"],
            # log_epoch_metrics=True,
        )
        self._log_hparams(wandb_logger)
        self._loggers = {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._task_args["logdir"]),
            "tensorboard": dl.TensorboardLogger(logdir=self._task_args["logdir"]),
            "wandb": wandb_logger
        }
        self._callbacks = [
            CheckpointCallback(
                logdir=self._task_args["logdir"],
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                topk=10,
            ),
        ]
        logging.info(f"logdir: {self._task_args['logdir']}")
        if isinstance(self._model, (Supervised, PIXL)):
            for source_id in self._model._encoder.keys():
                self._callbacks.append(
                    dl.AccuracyCallback(
                        input_key=f"logits_{source_id}",
                        target_key="targets",
                        log_on_batch=False,
                        prefix=f'source_{source_id}_'
                    )
                )
                self._callbacks.append(
                    dl.AUCCallback(
                        input_key=f"probs_{source_id}",
                        target_key="targets",
                        prefix=f'source_{source_id}_'
                    )
                )
                self._callbacks.append(
                    dl.ConfusionMatrixCallback(
                        input_key=f"logits_{source_id}",
                        target_key="targets",
                        num_classes=self._dataset.num_classes,
                        prefix=f'source_{source_id}_'
                    )
                )
        start_time = time.time()
        self._runner.train(
            model=self._model,
            criterion=self._criterion,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            loaders=self._dataset.get_all_loaders(),
            logdir=self._task_args["logdir"],
            num_epochs=self._task_args["num_epochs"],
            verbose=self._task_args["verbose"],
            # resume=self._task_args['resume'],
            timeit=self._task_args["timeit"],
            callbacks=self._callbacks,
            loggers=self._loggers,
            fp16=False,
        )
        print("--- Training took %s seconds ---" % ((time.time() - start_time) / self._task_args["num_epochs"]))
