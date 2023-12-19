from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10


def load_mnist_dataloaders(path, batch_size, transform):
    train_dataset = MNIST(path, train=True, download=True, transform=transform)
    test_dataset = MNIST(path, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def load_cifar10_dataloaders(path, batch_size, transform):
    train_dataset = CIFAR10(path, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(path, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
