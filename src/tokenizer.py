import torch
import torch.nn.functional as F
from diffusers import VQModel
from torchvision import transforms


class VQVAETokenizer:
    """
    A wrapper for a pretrained VQ-VAE model from Hugging Face diffusers.
    This class provides an interface to encode images into discrete tokens.
    """

    def __init__(
        self, vqvae_path: str = "CompVis/ldm-celebahq-256", device: str = "cpu"
    ):
        """
        Initializes the VQVAETokenizer.

        Args:
            vqvae_path (str): The path to the pretrained VQ-VAE model on Hugging Face Hub.
            device (str): The device to load the model on ('cpu' or 'cuda').
        """
        # self.vqvae = VQModel.from_pretrained(vqvae_path, subfolder="vqvae").eval()
        self.vqvae = (
            VQModel.from_pretrained(
                vqvae_path, subfolder="vqvae", sane_index_shape=True
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.num_channels = self.vqvae.config.in_channels
        self.vocab_size = self.vqvae.quantize.n_e

        # Preprocessing transforms
        self.resize = transforms.Resize(
            (self.vqvae.config.sample_size, self.vqvae.config.sample_size)
        )
        self.normalize = transforms.Normalize(
            mean=[0.5] * self.num_channels, std=[0.5] * self.num_channels
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of images to VQ-VAE token IDs.

        Supports:
        • uint8   tensors in [0,255]
        • float32 tensors in [0,1]  (common torchvision)
        • float32 tensors in [-1,1] (already pre-scaled)
        Returns
        -------
        (B, S) LongTensor
        """
        B = images.size(0)

        # ---- 1. dtype & basic scaling ----------------------------------------
        if images.dtype == torch.uint8:  # 0‥255 → 0‥1
            images = images.to(torch.float32).div_(255.0)
        elif images.amin() < -0.01:  # assume [-1,1] → 0‥1
            images = images.add_(1.0).div_(2.0)
        # else: already float 0‥1

        # ---- 2. channel fix for grayscale inputs -----------------------------
        if images.shape[1] != self.num_channels:
            repeat_factor = self.num_channels // images.shape[1]
            images = images.repeat(1, repeat_factor, 1, 1)

        # ---- 3. resize & single (0.5,0.5) normalisation ----------------------
        images = self.resize(images)  # (B, C, 256, 256)
        images = self.normalize(images)  # now in [-1,1]
        # ---- 4. VQ-VAE forward ----------------------------------------------
        latents = self.vqvae.encoder(images)
        _, _, (_, _, idx) = self.vqvae.quantize(latents)  # (B*H'*W',)
        return idx.reshape(B, -1).to(torch.long)  # (B, S)

    def get_vocab_size(self) -> int:
        """Returns the size of the VQ-VAE's codebook."""
        return self.vqvae.quantize.n_e

    def get_sequence_length(self) -> int:
        """
        Returns the actual sequence length that the tokenizer produces.

        This is determined by doing a dummy forward pass.
        """
        # Create a dummy input tensor
        dummy_input = torch.randn(
            1,
            self.num_channels,
            self.vqvae.config.sample_size,
            self.vqvae.config.sample_size,
        )
        # Encode the dummy input
        token_indices = self.encode(dummy_input)
        # Return the actual sequence length produced by the tokenizer
        return token_indices.shape[1]
