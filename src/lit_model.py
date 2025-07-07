import pytorch_lightning as pl
import torch
from torch.optim import AdamW

from src.model import GPT, GPTConfig


class LitGPT(pl.LightningModule):
    """
    PyTorch Lightning wrapper for our GPT model.

    This class handles the training, validation, and optimization logic,
    allowing us to keep the core model definition clean.
    """

    def __init__(self, model_config: GPTConfig, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = GPT(model_config)
        self.lr = lr

    def _prepare_batch(self, batch):
        """
        Prepares a batch for the model.
        The input from the dataloader is a tensor of shape (B, C, H, W).
        We flatten it to (B, S) where S = C * H * W to create a sequence.
        The input to the model is the sequence, and the target is the sequence shifted by one.
        """
        x, _ = batch  # We don't need the labels (y) for next-token prediction
        # Flatten the image into a sequence of pixels
        # Input shape: (B, C, H, W), e.g., (64, 1, 28, 28) for MNIST
        # Output shape: (B, S), e.g., (64, 784)
        sequences = x.view(x.size(0), -1)

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
