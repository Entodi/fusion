import copy
from gc import callbacks
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf, SCMode
import os
import optuna
import pandas as pd
import pickle
import seaborn as sns

from catalyst.utils.torch import load_checkpoint, unpack_checkpoint

from fusion.dataset.misc import SetId
from fusion.model import model_provider
from fusion.task.linear_evaluation import LinearEvaluationTask
from fusion.task.linear_evaluation import LinearEvaluationTaskBuilder

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import wandb

from typing import Sequence
from typing import Union


class CustomWeightsAndBiasesCallback(object):
    def __init__(
        self,
        metric_name: Union[str, Sequence[str]] = "value",
    ):
        self._metric_name = metric_name

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if isinstance(self._metric_name, str):
            if len(trial.values) > 1:
                # Broadcast default name for multi-objective optimization.
                names = ["{}_{}".format(self._metric_name, i) for i in range(len(trial.values))]

            else:
                names = [self._metric_name]

        else:
            if len(self._metric_name) != len(trial.values):
                raise ValueError(
                    "Running multi-objective optimization "
                    "with {} objective values, but {} names specified. "
                    "Match objective values and names, or use default broadcasting.".format(
                        len(trial.values), len(self._metric_name)
                    )
                )

            else:
                names = [*self._metric_name]

        metrics = {name: value for name, value in zip(names, trial.values)}
        attributes = {"direction": [d.name for d in study.directions]}

        wandb.config.update(attributes)
        wandb.log({**trial.params, **metrics}, step=trial.number)


class LogRegEvaluationTaskBuilder(LinearEvaluationTaskBuilder):
    def create_new_task(self, config: DictConfig, task_args: DictConfig, seed: int = 343):
        """
        Method to create new logistic regression evaluation task

        Args:
            task_args: dictionary with task's parameters from config
        """
        self._task = LogRegEvaluationTask(config, task_args.args, seed=seed)

    def add_model(self, model_config: DictConfig):
        """
        Method for add model to linear evaluation task
        
        Args:
            model_config: dictionary with model's parameters from config
        """
        self._task.model = {}
        # get number of classes
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
        # create linear evaluators
        for source_id, encoder in pretrained_model.get_encoder_list().items():
            encoder_extractor_args = {
                "encoder": encoder,
                "source_id": int(source_id),
            }
            print(encoder_extractor_args)
            encoder_extractor = model_provider.get(
                "EncoderExtractor", **encoder_extractor_args
            )
            self._task.model[source_id] = encoder_extractor

    def add_criterion(self, criterion_config: DictConfig):
        pass

    def add_optimizer(self, optimizer_config: DictConfig):
        pass

    def add_scheduler(self, scheduler_config: DictConfig):
        pass


class LogRegEvaluationTask(LinearEvaluationTask):
    def run(self):
        hparams = OmegaConf.to_container(
            self._config, structured_config_mode=SCMode.DICT_CONFIG, resolve=True)
        wandb.init(
            project=self._task_args["project"],
            name=self._task_args["name"],
            entity=self._task_args["entity"],
            config=hparams,
        )
        target_sources = list(self._dataset._target_to_source.keys())
        modifier = '_'.join(target_sources)
        for source_id in self._model.keys():
            self._reset_seed()
            results = []
            logdir = self._task_args["logdir"] + f"/logreg_{source_id}/{modifier}/"
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            print (logdir)
            model = None

            representations_targets = self._get_representation(source_id)
            scaled_rs_ts = self._rescale_representations(representations_targets)
            X_train, y_train = scaled_rs_ts[SetId.TRAIN]
            X_valid, y_valid = scaled_rs_ts[SetId.VALID]
            model = self._search_train(
                X_train, y_train, X_valid, y_valid, logdir, source_id)
            assert model is not None
            for set_name in [SetId.TRAIN, SetId.VALID, SetId.INFER]:
                scaled_representation, targets = scaled_rs_ts[set_name]
                metrics = self._evaluate(model, scaled_representation, targets)
                metrics = [source_id, set_name] + metrics
                results.append(metrics)

            if self._task_args['save_representation']:
                for set_name in [SetId.TRAIN, SetId.VALID, SetId.INFER]:
                    self._save_representation(
                        logdir, set_name, representations_targets[set_name], targets)
            self._save_metrics(results, logdir, source_id)
            print(results)

    @staticmethod
    def _rescale_representations(representations_targets):
        scaler = StandardScaler()
        for set_name in [SetId.TRAIN, SetId.VALID, SetId.INFER]:
            representation, _ = representations_targets[set_name]
            if set_name == SetId.TRAIN:
                scaler.fit(representation)
                scaled_representation = scaler.transform(representation)
            else:
                scaled_representation = scaler.transform(representation)
            representations_targets[set_name][0] = scaled_representation
        return representations_targets

    @staticmethod
    def _save_representation(logdir, set_name, representation, targets):
        data_dict = {
            'representation': representation,
            'targets': targets
        }
        with open(logdir + f'representation_{set_name}.pickle', 'wb') as handle:
            pickle.dump(
                data_dict,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def _save_metrics(self, results, logdir, source_id):
        columns = []
        if self._dataset.num_classes > 2:
            columns = [
                'Source Id', 'Set Name', 'ACC', 'BACC',
                'OVO ROCAUC Macro', 'OVR ROCAUC Macro',
                'OVO ROCAUC Weighted', 'OVR ROCAUC Weighted',
            ]
        elif self._dataset.num_classes == 2:
            columns = [
                'Source Id', 'Set Name', 'ACC',
                'BACC', 'ROCAUC', 'Brier'
            ]
        else:
            raise NotImplementedError
        assert len(columns) != 0
        results = pd.DataFrame(results, columns=columns)
        results.to_csv(logdir + 'metrics.csv', index=False)
        wandb.log({f"{logdir}": results})
        results.columns = [f'{source_id}_{col}' for col in results.columns]
        print (results.columns)
        wandb.config.update({**results.to_dict()})

    def _evaluate(self, model, representation, targets):
        predicted = model.predict(representation)
        probs = model.predict_proba(representation)
        acc = accuracy_score(targets, predicted)
        bacc = balanced_accuracy_score(targets, predicted)
        if self._dataset.num_classes > 2:
            ovo_ras_macro = roc_auc_score(
                targets, probs, multi_class='ovo', average='macro')
            ovr_ras_macro = roc_auc_score(
                targets, probs, multi_class='ovr', average='macro')
            ovo_ras_weighted = roc_auc_score(
                targets, probs, multi_class='ovo', average='weighted')
            ovr_ras_weighted = roc_auc_score(
                targets, probs, multi_class='ovr', average='weighted')
            results = [
                acc, bacc,
                ovo_ras_macro, ovr_ras_macro,
                ovo_ras_weighted, ovr_ras_weighted
            ]
        elif self._dataset.num_classes == 2:
            probs = probs[:, 1] # e.g. in this case only on AD
            ras = roc_auc_score(targets, probs)
            brier = brier_score_loss(targets, probs)
            results = [acc, bacc, ras, brier]
        else:
            raise NotImplementedError
        return results

    def _search_train(self, X_train, y_train, X_valid, y_valid, logdir, source_id):

        #multi_class = 'raise'
        #average = 'macro'
        #if self._dataset.num_classes > 2:
        #    multi_class = self._task_args['scorer']['multi_class']
        #    average = self._task_args['scorer']['average']

        """
        def custom_roc_auc_scorer(
            clf, X, y, multi_class=multi_class, average=average
        ):
            if multi_class == 'raise':
                preds = clf.predict_proba(X)[:, 1]
            else:
                preds = clf.predict_proba(X)
            return roc_auc_score(
                y,
                preds,
                multi_class=multi_class,
                average=average
            )
        """

        optuna_args = self._task_args["optuna"]
        solver = optuna_args['solver']
        num_trials = optuna_args["num_trials"]
        seed = optuna_args["seed"]
        scoring = sklearn.metrics.get_scorer(self._task_args['scoring'])

        def _objective(trial):
            C_ = trial.suggest_loguniform('C', 1e-6, 1e3)
            penalty_ = 'elasticnet'
            l1_ratio_ = trial.suggest_uniform('l1_ratio', 0.0, 1.0)
            clf = LogisticRegression(
                random_state=seed,
                class_weight='balanced',
                penalty=penalty_,
                C=C_,
                solver=solver,
                l1_ratio=l1_ratio_,
                fit_intercept=False,
                max_iter=200
            )
            clf.fit(X_train, y_train)
            score = scoring(clf, X_valid, y_valid)
            return score

        wandbc = CustomWeightsAndBiasesCallback(metric_name=self._task_args['scoring'])
        study = optuna.create_study(direction='maximize')
        study.optimize(_objective, n_trials=num_trials, callbacks=[wandbc], n_jobs=4)
        trial = study.best_trial
        print(trial)
        with open(logdir + 'best_trial_optuna.pickle', 'wb') as handle:
            pickle.dump(
                trial,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        clf = LogisticRegression(
            random_state=seed,
            class_weight='balanced',
            penalty='elasticnet',
            C=trial.params['C'],
            solver=solver,
            fit_intercept=False,
            l1_ratio=trial.params['l1_ratio'] if 'l1_ratio' in trial.params else None,
            max_iter=200
        )
        clf.fit(X_train, y_train)
        self._save_importance(clf, logdir, modifier=source_id)
        return clf

    def _get_representation(self, source_id):
        representations_targets = {}
        for set_name in [SetId.TRAIN, SetId.VALID, SetId.INFER]:
            predictions = self._runner.predict_loader(
                loader=self._dataset.get_loader(set_name),
                model=self._model[source_id]
            )
            representation = None
            targets = None
            for preds in predictions:
                batch_output, batch_targets = preds
                batch_representation = batch_output.z[int(source_id)]
                batch_targets = batch_targets.cpu().numpy()
                batch_representation = batch_representation.detach().cpu().numpy()
                representation = self._concat_x_to_y(
                    batch_representation, representation)
                targets = self._concat_x_to_y(batch_targets, targets)
            representations_targets[set_name] = [representation, targets]
        return representations_targets

    @staticmethod
    def _concat_x_to_y(x, y):
        y = np.concatenate(
            (y, x), axis=0) if y is not None else x
        return y

    @staticmethod
    def _save_importance(clf, logdir, modifier=''):
        importance = clf.coef_
        df = pd.DataFrame(importance)
        df = df.sort_values(
            by=0, ascending=False, axis=1)
        df.to_csv(logdir + f'{modifier}_importance.csv', index=True)
        plt.figure(figsize=(16, 3))
        sns.barplot(data=df)
        plt.tight_layout()
        plt.savefig(logdir + f'{modifier}_importance.png', dpi=600)
