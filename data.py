from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
import torch
import numpy as np


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


def whitening_transformation(gradient_matrix):
    # centering data - zero mean
    transformed_matrix = torch.complex(gradient_matrix - torch.mean(gradient_matrix, dim=0), torch.zeros(gradient_matrix.size()))

    # finding cov matrix
    covariance_matrix = transformed_matrix.T
    covariance_matrix = torch.cov(covariance_matrix)
    assert not ((torch.any(torch.isnan(covariance_matrix)).item()) or (torch.any(torch.isinf(covariance_matrix)).item())), \
        'There are NaN or Inf values in covariance matrix array'

    # eigen decomposition
    eigvals, eigvectors = torch.linalg.eig(covariance_matrix)
    # eigvals, eigvectors = eigvals.real, eigvectors.real
    # eigvals[eigvals < 0] = 1e-10
    # assert not ((torch.any(torch.isnan(eigvals)).item()) or (torch.any(torch.isinf(eigvals)).item())), \
    #     'There are NaN or Inf values in eigenvalue array'
    sqrt_eigvals = torch.sqrt(eigvals)
    # assert not ((torch.any(torch.isnan(sqrt_eigvals)).item()) or (torch.any(torch.isinf(sqrt_eigvals)).item())), 'There are NaN or Inf values in sqrt eigenvalue array'
    inverse_sqrt_eigvals = (1 / sqrt_eigvals)
    inverse_sqrt_eigvals[torch.isnan(inverse_sqrt_eigvals.real)] = 0.+0.j
    # assert not ((torch.any(torch.isnan(inverse_sqrt_eigvals)).item()) or (torch.any(torch.isinf(inverse_sqrt_eigvals)).item())), 'There are NaN or Inf values in inverse sqrt eigenvalue array'

    # pca whitening
    pca_whitening = (torch.diag(inverse_sqrt_eigvals) @ eigvectors.T @ transformed_matrix.T).T
    # pca_whitening = torch.nan_to_num(pca_whitening)
    # assert not ((torch.any(torch.isnan(pca_whitening)).item()) or (torch.any(torch.isinf(pca_whitening)).item())), 'There are NaN or Inf values in pca whitened array'

    # zca_whitening
    zca_whitening = (eigvectors @ torch.diag(inverse_sqrt_eigvals) @ eigvectors.T @ transformed_matrix.T).T
    # zca_whitening = torch.nan_to_num(zca_whitening)
    # assert not ((torch.any(torch.isnan(zca_whitening)).item()) or (torch.any(torch.isinf(zca_whitening)).item())), 'There are NaN or Inf values in zca whitened array'

    return pca_whitening, zca_whitening
