import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError
from lightning.pytorch import seed_everything
import os
from rich import print
from icecream import install
install()


@hydra.main(version_base=None, config_path='config', config_name='debug')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    try:
        if cfg.train_params.seed is not None:
            seed_everything(cfg.train_params.seed, workers=True)
    except ConfigAttributeError:
        pass
        
    model = instantiate(cfg.model)
    dm = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, dm)
    
if __name__ == '__main__':
    main()