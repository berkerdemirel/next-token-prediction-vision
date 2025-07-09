import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from src.lit_model import LitGPT


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    The main training script.

    This script is orchestrated by Hydra. It will instantiate the datamodule,
    model, and trainer based on the provided configuration files.
    """
    # --- Set seed for reproducibility --- #
    pl.seed_everything(cfg.seed)

    # --- Instantiate DataModule --- #
    datamodule = hydra.utils.instantiate(cfg.data)

    # --- Instantiate Model --- #
    # The model config is nested under the data config to get vocab_size and block_size
    model_config = hydra.utils.instantiate(cfg.model)
    model = LitGPT(model_config, use_vqvae=cfg.lit_model.use_vqvae, vqvae_path=cfg.lit_model.vqvae_path, lr=cfg.lit_model.lr)
    # --- Instantiate Trainer --- #
    # The trainer is now instantiated directly from the config, which includes the logger and callbacks
    trainer = hydra.utils.instantiate(cfg.trainer)

    # --- Start Training --- #
    trainer.fit(model, datamodule=datamodule)

    # --- Start Testing --- #
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
