from fusion.experiment import Experiment
import hydra
from omegaconf import DictConfig
import time


@hydra.main(config_path="./configs", config_name="default")
def my_experiment(cfg: DictConfig) -> None:
    exp = Experiment(cfg)
    exp.setup_new_experiment()
    exp.start()
    #print ('Pass')


if __name__ == "__main__":
    start_time = time.time()
    my_experiment()
    print("--- Whole experiment took %s seconds ---" % (time.time() - start_time))
