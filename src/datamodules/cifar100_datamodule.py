import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100
from torchvision import transforms


class CIFAR100DataModule(pl.LightningDataModule):
    """
    LightningDataModule for the CIFAR100 dataset.
    This module handles the downloading, splitting, and loading of the CIFAR100 dataset.
    The transform pipeline converts images to tensors.
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 4,
        val_split_size: int = 5000,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_size = val_split_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # For pixel-based approach, we might want to scale to 0-255 and cast to long.
                # For VQ-VAE, we just need a tensor. The model will handle normalization/tokenization.
                # lambda x: (x * 255).long(),
            ]
        )

    def prepare_data(self):
        """Downloads the CIFAR100 dataset if not already present."""
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        """
        Assigns train/val/test datasets.
        This method is called on every GPU in a distributed setting.
        """
        if stage == "fit" or stage is None:
            cifar100_full = CIFAR100(
                self.data_dir, train=True, transform=self.transform
            )
            train_size = len(cifar100_full) - self.val_split_size
            self.cifar100_train, self.cifar100_val = random_split(
                cifar100_full, [train_size, self.val_split_size]
            )

        if stage == "test" or stage is None:
            self.cifar100_test = CIFAR100(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        """Returns the training dataloader."""
        return DataLoader(
            self.cifar100_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Returns the validation dataloader."""
        return DataLoader(
            self.cifar100_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Returns the test dataloader."""
        return DataLoader(
            self.cifar100_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
