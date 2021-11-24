import torch
from crlapi.benchmark import StreamTrainer
import hydra
from omegaconf import DictConfig, OmegaConf


def to_dict(cfg):
    r = {}
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            td = to_dict(v)
            for kk in td:
                r[k + "/" + kk] = td[kk]
        else:
            r[k] = v
    return r

@hydra.main(config_path=".", config_name="test_finetune_mlp.yaml")
def main(cfg):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    import time

    stream_trainer = StreamTrainer()
    stream_trainer.run(cfg)

if __name__ == "__main__":
    main()
