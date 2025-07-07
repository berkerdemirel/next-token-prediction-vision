import pytest
import torch
from src.datamodules.mnist_datamodule import MNISTDataModule
from src.datamodules.cifar100_datamodule import CIFAR100DataModule

@pytest.fixture
def mnist_dm():
    return MNISTDataModule(data_dir="data/test_mnist", batch_size=4)

@pytest.fixture
def cifar100_dm():
    return CIFAR100DataModule(data_dir="data/test_cifar100", batch_size=4)

def test_mnist_datamodule_setup(mnist_dm):
    mnist_dm.prepare_data()
    mnist_dm.setup("fit")
    assert mnist_dm.mnist_train
    assert mnist_dm.mnist_val
    mnist_dm.setup("test")
    assert mnist_dm.mnist_test

def test_mnist_dataloaders(mnist_dm):
    mnist_dm.prepare_data()
    mnist_dm.setup()
    train_loader = mnist_dm.train_dataloader()
    val_loader = mnist_dm.val_dataloader()
    test_loader = mnist_dm.test_dataloader()
    
    x, y = next(iter(train_loader))
    assert x.shape == (4, 1, 28, 28)
    assert y.shape == (4,)
    assert x.dtype == torch.long
    
    x, y = next(iter(val_loader))
    assert x.shape == (4, 1, 28, 28)
    assert y.shape == (4,)
    assert x.dtype == torch.long

    x, y = next(iter(test_loader))
    assert x.shape == (4, 1, 28, 28)
    assert y.shape == (4,)
    assert x.dtype == torch.long

def test_cifar100_datamodule_setup(cifar100_dm):
    cifar100_dm.prepare_data()
    cifar100_dm.setup("fit")
    assert cifar100_dm.cifar100_train
    assert cifar100_dm.cifar100_val
    cifar100_dm.setup("test")
    assert cifar100_dm.cifar100_test

def test_cifar100_dataloaders(cifar100_dm):
    cifar100_dm.prepare_data()
    cifar100_dm.setup()
    train_loader = cifar100_dm.train_dataloader()
    val_loader = cifar100_dm.val_dataloader()
    test_loader = cifar100_dm.test_dataloader()

    x, y = next(iter(train_loader))
    assert x.shape == (4, 3, 32, 32)
    assert y.shape == (4,)
    assert x.dtype == torch.long

    x, y = next(iter(val_loader))
    assert x.shape == (4, 3, 32, 32)
    assert y.shape == (4,)
    assert x.dtype == torch.long

    x, y = next(iter(test_loader))
    assert x.shape == (4, 3, 32, 32)
    assert y.shape == (4,)
    assert x.dtype == torch.long
