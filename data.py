from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import MNIST, CIFAR10
import numpy as np
import matplotlib.colors as colors
import torch


def load_mnist_dataloaders(path, batch_size, transform):
    train_dataset = MNIST(path, train=True, download=True, transform=transform)
    test_dataset = MNIST(path, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def load_cifar10_dataloaders(path, batch_size, transform, batch_sampler=False):
    train_dataset = CIFAR10(path, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(path, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if not batch_sampler:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    else:
        high_hsv_sampler = HighHSVBatchSampler(test_dataset, batch_size)
        test_loader = DataLoader(test_dataset, batch_sampler=high_hsv_sampler)

    return train_loader, test_loader


def _demean(input_matrix):
    # Note that this is the same as J = (I - 11^T/n)
    ones = np.ones((input_matrix.shape[0], 1))
    J = np.identity(input_matrix.shape[0]) - ((np.matmul(ones, ones.T)) / input_matrix.shape[0])
    return np.matmul(J, input_matrix)


# whitening implementation taken from https://courses.cs.washington.edu/courses/cse446/19au/section8_SVD.html
class WhiteningTransformation(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def project(self, input_matrix):
        # Note that this is the same as V
        return np.matmul(input_matrix, self.V.T)

    def scale(self, input_matrix):
        # Note that this is the same as S^{-1}
        return np.multiply(input_matrix, (1 / self.S))

    def unrotate(self, input_matrix):
        # Note that this is the same as V^{T}
        return np.matmul(input_matrix, self.V)

    def transform(self, input_matrix):
        demean_matrix = _demean(input_matrix)
        self.U, self.S, self.V = np.linalg.svd(demean_matrix)
        projected_matrix = self.project(demean_matrix)
        scaled_matrix = self.scale(projected_matrix)
        unrotated_matrix = self.unrotate(scaled_matrix)
        return unrotated_matrix


class HighHSVBatchSampler(Sampler):
    def __init__(self, data, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.data = data

        sv_arr = []
        for elem, _ in self.data:
            rgb = np.asarray(elem).transpose(1, 2, 0)
            sv = colors.rgb_to_hsv(rgb)[:, :, 1:]
            sv = np.mean(sv)
            sv_arr.append(sv)
        self.batch_idxs = torch.chunk(reversed(torch.argsort(torch.tensor(sv_arr))), len(self))

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for batch in self.batch_idxs:
            yield batch.tolist()
