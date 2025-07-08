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

    def _prepare_batch(self, batch, *, max_windows_per_image: int = 8):
        """
        Tokenise images and build (input, target) windows for next-token prediction.

        Each image contributes up to `max_windows_per_image` random windows.
        If the sequence is shorter than or equal to `block_size`, we return a single pair.
        """
        images, _ = batch

        # --- 1. Tokenise --------------------------------------------------------
        if self.use_vqvae:
            if self.tokenizer is None:
                raise RuntimeError("VQ-VAE tokenizer not initialised; call setup() first.")
            seq = self.tokenizer.encode(images)              # (B, S)
        else:
            seq = (images.view(images.size(0), -1)           # (B, S)
                        .mul(255).round()
                        .to(torch.long))

        seq = seq.to(self.device)
        B, S = seq.shape
        K = self.model.block_size          # context length

        # --- 2. Short sequences -------------------------------------------------
        if S <= K:                         # no windowing needed
            return seq[:, :-1], seq[:, 1:]

        # --- 3. Build all windows as a single strided view ----------------------
        # windows_all : (B, num_windows, K+1)
        num_windows = S - K
        windows_all = seq.unfold(1, K + 1, 1)                # zero-copy view

        # --- 4. Randomly sample ≤ max_windows_per_image per sample -------------
        if num_windows > max_windows_per_image:
            # indices: (B, max_windows_per_image) without replacement
            idx = torch.arange(num_windows, device=seq.device)
            idx = idx.expand(B, num_windows)
            rand = torch.rand_like(idx.float())              # same shape
            perm = rand.argsort(dim=1)                       # random permutation
            idx = idx.gather(1, perm[:, :max_windows_per_image])
            # Gather needs an extra dim for broadcasting
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, K + 1)
            windows = windows_all.gather(1, idx_exp)         # (B, M, K+1)
        else:
            windows = windows_all                            # keep them all

        # --- 5. Split into inputs / targets and flatten batch ------------------
        # windows is (B, M, K+1) where M ≤ max_windows_per_image
        inputs  = windows[..., :-1].reshape(-1, K)           # (B*M, K)
        targets = windows[..., 1: ].reshape(-1, K)           # (B*M, K)

        return inputs, targets

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
