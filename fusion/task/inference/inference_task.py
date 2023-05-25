import copy
import os
from omegaconf import DictConfig
import pandas as pd
import logging

from catalyst import dl
from catalyst.utils.torch import load_checkpoint, unpack_checkpoint

from fusion.dataset import dataset_provider
from fusion.model import model_provider
from fusion.model.supervised import Supervised
from fusion.model.pixl import PIXL
from fusion.model.linear_evaluator import LinearEvaluator
from fusion.task.logreg_evaluation import LogRegEvaluationTask, \
    LogRegEvaluationTaskBuilder


class InferenceTaskBuilder(LogRegEvaluationTaskBuilder):
    def create_new_task(
        self, config: DictConfig, task_args: DictConfig, seed: int = 343
    ):
        self._task = InferenceTask(config, task_args.args, seed=seed)

    def add_model(self, model_config: DictConfig):
        num_classes = self._task.dataset._num_classes
        if "num_classes" in model_config.args.keys():
            if model_config.args["num_classes"] is None:
                model_config.args["num_classes"] = num_classes
        pretrained_checkpoint = model_config.args.pretrained_checkpoint
        # create model
        model_args = copy.deepcopy({**model_config.args})
        model_args.pop("pretrained_checkpoint")
        print (model_args)
        pretrained_model = model_provider.get(model_config.name, **model_args)
        # load checkpoint
        print (pretrained_checkpoint)
        checkpoint = load_checkpoint(pretrained_checkpoint)
        print (checkpoint.keys())
        checkpoint = {'model_state_dict': checkpoint}
        unpack_checkpoint(checkpoint, pretrained_model)
        self._task.model = pretrained_model


class InferenceTask(LogRegEvaluationTask):
    def run(self):
        logging.info(f"logdir: {self._task_args['logdir']}")
        assert isinstance(
            self._model, (
                LinearEvaluator, Supervised, PIXL
            )
        )
        self._callbacks = []
        print (self._dataset._sources)
        for source_id in self._dataset._sources:
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
        results = []
        for i, (set_name, loader) in enumerate(self._dataset.get_all_loaders().items()):
            metrics = self._runner.evaluate_loader(
                loader=loader,
                model=self._model,
                callbacks=self._callbacks,
                verbose=False
            )
            temp = pd.DataFrame(metrics, index=[i])
            temp['Set name'] = set_name
            results.append(temp)
        results = pd.concat(results, axis=0)
        results = results.drop(columns=['loss'])
        print (results)
        target_sources = list(self._dataset._target_to_source.keys())
        modifier = '_'.join(target_sources)
        results.to_csv(f'{self._task_args["logdir"]}/{modifier}_metrics.csv', index=False)
