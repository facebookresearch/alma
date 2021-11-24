import torch
from crlapi.benchmark import StreamTrainer
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path=".", config_name="test_finetune_mnist.yaml")
def main(cfg):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    import time

    print(cfg)
    print(dict(cfg))

    stream_trainer = StreamTrainer()
    stream_trainer.run(cfg)

if __name__ == "__main__":
    main()
