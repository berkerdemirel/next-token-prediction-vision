import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from src.lit_model import LitGPT
from src.tokenizer import VQVAETokenizer


class LitEvaluator(pl.LightningModule):
    """
    A PyTorch Lightning module for evaluating a pretrained GPT model.
    This module loads a checkpoint, freezes the GPT model, and trains
    linear heads for downstream tasks like classification and reconstruction.
    """

    def __init__(
        self,
        gpt_model: LitGPT,
        evaluate_classification: bool = True,
        evaluate_reconstruction: bool = True,
        num_classes: int = 100,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load the pretrained LitGPT model
        self.gpt_model = gpt_model
        # Freeze the GPT model
        self.gpt_model.freeze()

        self.n_embd = self.gpt_model.model.n_embd  # 768 for GPT-2-small

        # ---------- 2. Chunk-level aggregator ------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.chunk_attn = nn.MultiheadAttention(
            embed_dim=self.n_embd, num_heads=8, batch_first=True
        )
        self.chunk_ln = nn.LayerNorm(self.n_embd)

        # ---------- 3. Classification head --------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, num_classes),
        )

        # optimiser hyper-params
        self.lr = lr

    def setup(self, stage: str = None):
        """Move the tokenizer to the correct device."""
        if self.gpt_model.hparams.use_vqvae:
            if self.gpt_model.tokenizer:
                self.gpt_model.tokenizer.to(self.device)

    def training_step(self, batch, batch_idx):
        """Training step for the classification head."""
        if self.hparams.evaluate_classification:
            return self._train_classification(batch)
        return None

    def validation_step(self, batch, batch_idx):
        """Validation step for the classification head."""
        if self.hparams.evaluate_classification:
            self._test_classification(batch, prefix="val")

        # if self.hparams.evaluate_reconstruction:
        #     self._test_reconstruction(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        if self.hparams.evaluate_classification:
            self._test_classification(batch, prefix="test")

        # if self.hparams.evaluate_reconstruction:
        #     self._test_reconstruction(batch, batch_idx, prefix="test")

    def load_model(self, checkpoint_path: str) -> LitGPT:
        """Load a LitGPT model from a checkpoint, correctly handling dynamic vocab size."""
        # Load hyperparameters from the checkpoint
        hparams = torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
            "hyper_parameters"
        ]

        # Determine vocab_size
        if hparams["use_vqvae"]:
            # We need to instantiate a tokenizer to get the vocab size
            tokenizer = VQVAETokenizer(hparams["vqvae_path"])
            vocab_size = tokenizer.vocab_size
        else:
            vocab_size = 256

        # Update model_config with the correct vocab_size
        model_config = hparams["model_config"]
        model_config.vocab_size = vocab_size

        # Create a new model instance with the correct config
        model = LitGPT(model_config=model_config)
        # Initialise weights for layers that depend on vocab_size
        model.model._init_weights()

        # Load the state dict from the checkpoint
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )["state_dict"]
        model.load_state_dict(state_dict, strict=True)

        return model

    def _train_classification(self, batch):
        imgs, tgt = batch
        # idx, targets = self.gpt_model._prepare_batch(batch)
        # _, loss = self.gpt_model.forward(idx, targets)
        # print(loss)
        # breakpoint()
        # --- 3.1 Tokenise -------------------------------------------------
        if self.gpt_model.hparams.use_vqvae:
            if self.gpt_model.tokenizer is None:
                raise RuntimeError("VQ-VAE tokenizer not initialised.")
            idx = self.gpt_model.tokenizer.encode(imgs)
        else:
            idx = imgs.view(imgs.size(0), -1).mul(255).round().long()

        idx = idx.to(self.device)

        # --- 3.2 Frozen GPT features -------------------------------------
        with torch.no_grad():
            feats = (
                self.gpt_model.model.get_features_long(idx)
                if idx.size(1) > self.gpt_model.model.block_size
                else self.gpt_model.model.get_features(idx)  # (B, C)
            )

        # Guarantee shape (B, n_chunks, C)
        if feats.dim() == 2:  # one chunk only
            feats = feats.unsqueeze(1)  # (B, 1, C)

        # --- 3.3 Mean pooling -------------------------------------------
        # x = feats.mean(dim=1)  # (B, C)
        # --- 3.3 Attention pooling --------------------------------------
        B = feats.size(0)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat([cls, feats], dim=1)  # (B, 1+n, C)
        x, _ = self.chunk_attn(x, x, x)  # (B, 1+n, C)
        x = self.chunk_ln(x[:, 0])  # (B, C)

        # --- 3.4 Classify -----------------------------------------------
        logits = self.classifier(x)  # (B, num_classes)
        loss = F.cross_entropy(logits, tgt)
        acc = (logits.argmax(1) == tgt).float().mean()

        self.log("train_class_loss", loss, prog_bar=True)
        self.log("train_class_acc", acc, prog_bar=True)
        return loss

    def _test_classification(self, batch, prefix="test"):
        imgs, y = batch
        # -------- 1. tokenise -------------------------------------------------
        if self.gpt_model.hparams.use_vqvae:
            if self.gpt_model.tokenizer is None:
                raise RuntimeError("VQ-VAE tokenizer not initialised.")
            idx = self.gpt_model.tokenizer.encode(imgs)
        else:
            idx = imgs.view(imgs.size(0), -1).mul(255).round().long()

        idx = idx.to(self.device)

        # -------- 2. frozen-GPT features  (B, n_chunks?, C) -------------------
        with torch.no_grad():
            feats = (
                self.gpt_model.model.get_features_long(idx)
                if idx.size(1) > self.gpt_model.model.block_size
                else self.gpt_model.model.get_features(idx)  # (B, C)
            )

        if feats.dim() == 2:  # single chunk case
            feats = feats.unsqueeze(1)  # → (B, 1, C)

        # -------- 3. CLS-attention pooling  (B, C) ----------------------------
        B = feats.size(0)
        cls = self.cls_token.expand(B, 1, -1)  # (B, 1, C)
        cat = torch.cat([cls, feats], dim=1)  # (B, 1+N, C)
        x, _ = self.chunk_attn(cat, cat, cat)  # (B, 1+N, C)
        x = self.chunk_ln(x[:, 0])  # (B, C)

        # -------- 4. classify & log -------------------------------------------
        logits = self.classifier(x)  # (B, num_classes)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()

        self.log(f"{prefix}_class_loss", loss, prog_bar=True)
        self.log(f"{prefix}_class_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        """
        Optimise only the lightweight pooling-and-classifier head.
        GPT weights stay frozen.
        """
        trainable = []

        # CLS token is an nn.Parameter
        trainable.append(self.cls_token)

        # Attention pooling & layer-norm
        trainable += list(self.chunk_attn.parameters())
        trainable += list(self.chunk_ln.parameters())

        # Final MLP classifier
        trainable += list(self.classifier.parameters())

        # Safety: remove any param that is frozen or duplicated
        trainable = [p for p in trainable if p.requires_grad]

        if not trainable:  # nothing to train
            return None

        return AdamW(trainable, lr=self.hparams.lr)

    # def on_train_epoch_start(self):
    #     self._w0 = self.classifier[0].weight.clone()

    # def on_train_epoch_end(self, *a):
    #     delta = (self.classifier[0].weight - self._w0).abs().max()
    #     print("🔄 ΔW:", delta.item())  # must be > 1e-5
