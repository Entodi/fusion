from omegaconf import DictConfig, OmegaConf
import torch

from fusion.task import TaskDirector, task_builder_provider


class Experiment:
    # Singleton
    # To have global within experiments arguments
    def __init__(self, config: DictConfig):
        """
        Args:
            config:
        """
        print(OmegaConf.to_yaml(config))
        self._config = config["experiment"]
        self._task = None
        self._seed = self._config["seed"]

    def setup_new_experiment(self):
        torch.autograd.set_detect_anomaly(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        #torch.use_deterministic_algorithms(True)
        task_builder = task_builder_provider.get(self._config.task.name)
        task_director = TaskDirector(task_builder, self._config, self._seed)
        task_director.construct_task()
        self._task = task_director.get_task()

    def start(self):
        self._task.run()
