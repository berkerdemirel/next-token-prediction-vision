import pytorch_lightning as pl
import torch
from torch.optim import AdamW

from src.model import GPT, GPTConfig
from src.tokenizer import VQVAETokenizer


class LitGPT(pl.LightningModule):
    """
    PyTorch Lightning wrapper for our GPT model.
    This class handles the training, validation, and optimization logic,
    and orchestrates the tokenization of images.
    """

    def __init__(
        self,
        model_config: GPTConfig,
        lr: float = 1e-4,
        use_vqvae: bool = False,
        vqvae_path: str = "CompVis/vq-gan-imagenet-f16-16384",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT(model_config)
        self.lr = lr
        self.use_vqvae = use_vqvae
        self.vqvae_path = vqvae_path
        self.tokenizer = None  # To be initialized on the correct device in setup

    def setup(self, stage: str = None):
        """Initialize the tokenizer on the correct device."""
        if self.use_vqvae:
            self.tokenizer = VQVAETokenizer(self.vqvae_path, device=self.device)

    def _prepare_batch(self, batch):
        """
        Prepares a batch for the model.
        If using VQ-VAE, it tokenizes the images.
        Otherwise, it flattens the pixels into a sequence.
        """
        images, _ = batch  # We don't need the labels (y) for next-token prediction

        if self.use_vqvae:
            if self.tokenizer is None:
                raise RuntimeError("VQ-VAE tokenizer has not been initialized. Did you forget to call setup()?")
            # Tokenize images into a sequence of discrete tokens
            # Input shape: (B, C, H, W) -> Output shape: (B, S)
            sequences = self.tokenizer.encode(images)
        else:
            # Fallback to raw pixels
            # Input shape: (B, C, H, W) -> Output shape: (B, S)
            # Also, scale pixels to 0-255 and cast to long
            sequences = (images.view(images.size(0), -1) * 255).long()

        # The input to the model is the sequence from the beginning to the second to last token
        idx = sequences[:, :-1]
        # The target for the model is the sequence from the second token to the end
        targets = sequences[:, 1:]

        return idx, targets

    def forward(self, idx, targets=None):
        return self.model(idx, targets)

    def training_step(self, batch, batch_idx):
        idx, targets = self._prepare_batch(batch)
        _, loss = self.forward(idx, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        idx, targets = self._prepare_batch(batch)
        _, loss = self.forward(idx, targets)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
