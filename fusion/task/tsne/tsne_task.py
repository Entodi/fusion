from catalyst.utils.torch import load_checkpoint, unpack_checkpoint
import copy
import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from omegaconf import DictConfig
from omegaconf import DictConfig, OmegaConf, SCMode
import os
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import torch

from fusion.dataset.misc import SetId
from fusion.model import model_provider
from fusion.task.logreg_evaluation import LogRegEvaluationTask, \
    LogRegEvaluationTaskBuilder

import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


class TsneTaskBuilder(LogRegEvaluationTaskBuilder):
    def create_new_task(self, config: DictConfig, task_args: DictConfig, seed: int = 343):
        """
        Method to create new TSNE task

        Args:
            task_args: dictionary with task's parameters from config
        """
        self._task = TsneTask(config, task_args.args, seed=seed)

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
            model_config.args["num_classes"] = num_classes
        pretrained_checkpoint = model_config.args.pretrained_checkpoint
        # create model
        model_args = copy.deepcopy({**model_config.args})
        model_args.pop("pretrained_checkpoint")
        pretrained_model = model_provider.get(model_config.name, **model_args)
        # load checkpoint
        checkpoint = load_checkpoint(pretrained_checkpoint)
        checkpoint = {'model_state_dict': checkpoint}
        unpack_checkpoint(checkpoint, pretrained_model)
        # create linear evaluators
        for source_id, encoder in pretrained_model.get_encoder_list().items():
            encoder_extractor_args = {
                "encoder": encoder,
                "source_id": int(source_id),
            }
            encoder_extractor = model_provider.get(
                "EncoderExtractor", **encoder_extractor_args
            )
            self._task.model[source_id] = encoder_extractor


class TsneTask(LogRegEvaluationTask):
    def run(self):
        hparams = OmegaConf.to_container(self._config, structured_config_mode=SCMode.DICT_CONFIG, resolve=True)
        wandb.init(
            project=self._task_args["project"],
            name=self._task_args["name"],
            entity=self._task_args["entity"],
            config=hparams,
        )
        sns.set_style('whitegrid')
        for source_id in self._model.keys():
            tsne = TSNE(**self._task_args['tsne_args'])
            fig = Figure(figsize=(10, 10), dpi=300)
            canvas = FigureCanvas(fig)

            self._reset_seed()
            logdir = self._task_args["logdir"] + f"/tsne_{source_id}/"
            if not os.path.exists(logdir):
                os.makedirs(logdir)

            representations_targets = self._get_representation(source_id)
            scaled_rs_ts = self._rescale_representations(representations_targets)
            tsne_projections, all_targets = scaled_rs_ts[SetId.INFER]
            tsne_projections = tsne.fit_transform(tsne_projections)
            vis_x = tsne_projections[:, 0]
            vis_y = tsne_projections[:, 1]
            ax = fig.gca()
            df = {'Dimension 1': vis_x, 'Dimension 2': vis_y, 'Label': all_targets}
            df = pd.DataFrame.from_dict(df)
            try:
                df['Label'] = df['Label'].replace(self._dataset.label_2_name)
            except:
                pass
            df = df.sort_values(by=['Label'], ascending=False).reset_index(drop=True)
            # cmap = sns.color_palette("coolwarm", self._dataset.num_classes)
            cmap_list = list(sns.color_palette("coolwarm", self._dataset.num_classes))
            cmap_list_reverse = cmap_list[:self._dataset.num_classes//2] + cmap_list[-self._dataset.num_classes//2:][::-1]
            cmap = sns.color_palette(cmap_list_reverse)
            sns.set_style('whitegrid')
            sns.scatterplot(
                data=df, x='Dimension 1', y='Dimension 2', hue='Label', ax=ax, palette=cmap, s=80)
            ax.axis('off')
            ax.legend(ncol=2)
            canvas.draw()
            #plt.savefig(f'{logdir}/tsne_{source_id}.png', dpi=300, bbox_inches='tight')
            #plt.savefig(f'{logdir}/tsne_{source_id}.svg', dpi=300, bbox_inches='tight')
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = torch.FloatTensor(image).permute(2, 0, 1)
            image = wandb.Image(image, caption=f"TSNE_{source_id}")
            wandb.log({f"TSNE_{source_id}": image})
            canvas.print_figure(f'{logdir}/tsne_{source_id}.png', dpi=300, bbox_inches='tight')
            canvas.print_figure(f'{logdir}/tsne_{source_id}.svg', dpi=300, bbox_inches='tight')
