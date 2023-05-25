import copy
from omegaconf import DictConfig
import pickle
from tqdm import tqdm

from catalyst.utils.torch import load_checkpoint, unpack_checkpoint

from fusion.dataset.misc import SetId
from fusion.model import model_provider
from fusion.task.logreg_evaluation import LogRegEvaluationTask, \
    LogRegEvaluationTaskBuilder


class FeatureExtractionTaskBuilder(LogRegEvaluationTaskBuilder):
    def create_new_task(
        self,
        config: DictConfig,
        task_args: DictConfig,
        seed: int = 343
    ):
        self._task = FeatureExtractionTask(config, task_args.args, seed=seed)


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
        keys = checkpoint.copy().keys()
        for key in keys:
            if key.startswith("_latent_heads"):
                checkpoint.pop(key)
            if key.startswith("_conv_heads"):
                checkpoint.pop(key)
        #checkpoint = {'model_state_dict': checkpoint}
        check_keys = pretrained_model.load_state_dict(checkpoint, strict=False)
        for mkey in check_keys.missing_keys:
            if mkey.startswith("encoder"):
                assert False
        for mkey in check_keys.unexpected_keys:
            if mkey.startswith("encoder"):
                assert False
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


class FeatureExtractionTask(LogRegEvaluationTask):
    def run(self):
        for idx, source_id in enumerate(self._model.keys()):
            self._reset_seed()
            self._get_representation(source_id, idx)

    def _get_representation(self, source_id, idx):
        target_sources = list(self._dataset._target_to_source.keys())
        modifier = '_'.join(target_sources)
        logdir = self._task_args["logdir"]
        for set_name in [SetId.INFER]:#SetId.TRAIN, SetId.VALID,
            predictions = self._runner.predict_loader(
                loader=self._dataset.get_loader(set_name),
                model=self._model[source_id]
            )
            for batch_idx, preds in tqdm(enumerate(predictions)):
                representations = {}
                batch_output, _ = preds
                for k in batch_output.attrs['latents'].keys():
                    representations[k] = batch_output.attrs['latents'][k].cpu().numpy()
                filename = f'{logdir}/{modifier}_representation_{self._dataset._sources[idx]}_{set_name}_{batch_idx}.pickle'
                with open(filename, 'wb') as handle:
                    pickle.dump(
                        representations,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL
                    )
