import torch
from diffusers import VQModel
from torchvision import transforms
import torch.nn.functional as F

class VQVAETokenizer:
    """
    A wrapper for a pretrained VQ-VAE model from Hugging Face diffusers.
    This class provides an interface to encode images into discrete tokens.
    """

    def __init__(self, vqvae_path: str = "CompVis/ldm-celebahq-256", device: str = "cpu"):
        """
        Initializes the VQVAETokenizer.

        Args:
            vqvae_path (str): The path to the pretrained VQ-VAE model on Hugging Face Hub.
            device (str): The device to load the model on ('cpu' or 'cuda').
        """
        # self.vqvae = VQModel.from_pretrained(vqvae_path, subfolder="vqvae").eval()
        self.vqvae = VQModel.from_pretrained(vqvae_path, subfolder="vqvae").to(device).eval()
        self.device = device
        self.num_channels = self.vqvae.config.in_channels

        # Preprocessing transforms
        self.resize = transforms.Resize((self.vqvae.config.sample_size, self.vqvae.config.sample_size))
        self.normalize = transforms.Normalize(mean=[0.5] * self.num_channels, std=[0.5] * self.num_channels)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of images into sequences of token IDs.

        Args:
            images (torch.Tensor): A batch of images with shape (B, C, H, W).

        Returns:
            torch.Tensor: A batch of token ID sequences with shape (B, sequence_length).
        """
        # The VQ-VAE might expect a different number of input channels
        if images.shape[1] != self.num_channels:
            # This is a simple way to handle grayscale to RGB, might need adjustment
            images = images.repeat(1, self.num_channels, 1, 1)

        processed_images = self.resize(images)
        processed_images = self.normalize(processed_images)
        # processed_images = processed_images.to(self.device)

        # Encode the images into latent representations
        latents = self.vqvae.encoder(processed_images)
        # Quantize the latents to get token indices
        _, _, [_, _, token_indices] = self.vqvae.quantize(latents)

        # Flatten the token indices from (B, H, W) to (B, H*W)
        return token_indices.view(token_indices.shape[0], -1)

    def get_vocab_size(self) -> int:
        """Returns the size of the VQ-VAE's codebook."""
        return self.vqvae.quantize.num_embeddings

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
