import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

class MNISTDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the MNIST dataset.

    This module handles the downloading, splitting, and loading of the MNIST dataset.
    For the initial approach, we will treat the pixel values directly as a sequence.
    The transform pipeline converts images to tensors, which can then be flattened 
    in the model to create a sequence of pixels.
    """
    def __init__(self, data_dir: str = "data/", batch_size: int = 64, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # Note: To treat pixels as discrete tokens from 0-255, we will not normalize here.
        # The model will receive tensors of shape (B, 1, 28, 28) with values from 0-255.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: (x * 255).long() # Scale to 0-255 and convert to long for embedding
        ])

    def prepare_data(self):
        """Downloads the MNIST dataset if not already present."""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        """
        Assigns train/val/test datasets.
        This method is called on every GPU in a distributed setting.
        """
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        """Returns the training dataloader."""
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """Returns the validation dataloader."""
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        """Returns the test dataloaloader."""
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)
