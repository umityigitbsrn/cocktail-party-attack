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


# WARNING: This implementation is not for this project, while mentioning whitening in the paper this whitening is not
#  meant
class WhiteningTransformationV2(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def rotate(self, input_matrix):
        return np.dot(input_matrix, self.U)

    def whiten(self, input_matrix):
        return input_matrix / np.sqrt(self.S + 1e-5)

    def transform(self, input_matrix):
        demean_matrix = _demean(input_matrix)
        cov = np.dot(demean_matrix.T, demean_matrix) / demean_matrix.shape[0]
        self.U, self.S, self.V = np.linalg.svd(cov)
        rotated = self.rotate(demean_matrix)
        whitened = self.whiten(rotated)
        return whitened


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
        hsv_sorted_img_idxs = reversed(torch.argsort(torch.tensor(sv_arr)))
        hsv_sorted_img_chunked_idxs = list(torch.chunk(hsv_sorted_img_idxs, batch_size))
        hsv_sorted_img_chunked_len_arr = [len(x) for x in hsv_sorted_img_chunked_idxs]
        hsv_sorted_img_chunked_pointer_arr = [0 for _ in hsv_sorted_img_chunked_idxs]
        pointer = 0
        idxs = []
        done = [False for _ in hsv_sorted_img_chunked_idxs]
        while not all(done):
            if hsv_sorted_img_chunked_pointer_arr[pointer] == hsv_sorted_img_chunked_len_arr[pointer] - 1:
                done[pointer] = True
                pointer = (pointer + 1) % len(hsv_sorted_img_chunked_idxs)
                continue

            idxs.append(hsv_sorted_img_chunked_idxs[pointer][hsv_sorted_img_chunked_pointer_arr[pointer]])
            hsv_sorted_img_chunked_pointer_arr[pointer] += 1
            pointer = (pointer + 1) % len(hsv_sorted_img_chunked_idxs)

        self.batch_idxs = torch.chunk(torch.tensor(idxs), len(self))

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for batch in self.batch_idxs:
            yield batch.tolist()
