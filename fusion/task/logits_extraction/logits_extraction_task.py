from omegaconf import DictConfig
import pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F

from fusion.dataset.misc import SetId
from fusion.task.inference import InferenceTaskBuilder
from fusion.task import ATask


class LogitsExtractionTaskBuilder(InferenceTaskBuilder):
    def create_new_task(
        self, config: DictConfig, task_args: DictConfig, seed: int = 343
    ):
        self._task = LogitsExtractionTask(config, task_args.args, seed=seed)


class LogitsExtractionTask(ATask):
    def run(self):
        self._get_predictions()

    def _get_predictions(self):
        target_sources = list(self._dataset._target_to_source.keys())
        modifier = '_'.join(target_sources)
        logdir = self._task_args["logdir"]
        for set_name in [SetId.TRAIN, SetId.VALID, SetId.INFER]:
            logits = self._runner.predict_loader(
                loader=self._dataset.get_loader(set_name),
                model=self._model
            )
            predictions = {}
            targets = []
            for logit in tqdm(logits):
                batch_output, y = logit
                z = batch_output.z
                for source_id in z.keys():
                    probs = F.softmax(z[source_id], dim=1)
                    preds = torch.argmax(probs, dim=1)
                    if source_id not in predictions:
                        predictions[source_id] = []
                    preds = preds.cpu().numpy()
                    predictions[source_id].append(preds)
                y = y.cpu().numpy()
                targets.append(y)

            for idx, source_id in enumerate(predictions.keys()):
                filename = f'{logdir}/{modifier}_predictions_{self._dataset._sources[idx]}_{set_name}.pickle'
                with open(filename, 'wb') as handle:
                    pickle.dump(
                        predictions[source_id],
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
                filename = f'{logdir}/{modifier}_targets_{self._dataset._sources[idx]}_{set_name}.pickle'
                with open(filename, 'wb') as handle:
                    pickle.dump(
                        targets,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )

