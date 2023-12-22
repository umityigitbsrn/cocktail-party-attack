import torch.nn as nn
import torch
import torch.nn.functional as F
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


class NonGaussianityLoss(nn.Module):
    def __init__(self, *args, alpha_param=1, **kwargs):
        super(NonGaussianityLoss, self).__init__(*args, **kwargs)
        # the alpha param should be in [1, 2] - [Independent component analysis: algorithms and applications]
        self.alpha_param = alpha_param

    def forward(self, inputs):
        # elementwise operations
        out_score = torch.log(torch.cosh(self.alpha_param * inputs) ** 2) / (self.alpha_param ** 2)
        out_score = torch.mean(out_score, dim=1)
        return -1 * torch.mean(out_score)


class TotalVariationLoss(nn.Module):
    def __init__(self, input_height, input_width, input_channel, *args, **kwargs):
        super(TotalVariationLoss, self).__init__(*args, **kwargs)
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel

    def forward(self, inputs):
        out_score = inputs.reshape(inputs.size(0), self.input_channel, self.input_height, self.input_width)
        first_score = torch.mean(torch.abs(out_score[:, :, :, :-1] - out_score[:, :, :, 1:]))
        second_score = torch.mean(torch.abs(out_score[:, :, :-1, :] - out_score[:, :, :1, :]))
        return first_score + second_score


class MutualIndependenceLoss(nn.Module):
    def __init__(self, t_param, *args, **kwargs):
        super(MutualIndependenceLoss, self).__init__(*args, **kwargs)
        self.t_param = t_param

    def forward(self, inputs):
        out_score = pairwise_cosine_similarity(inputs)
        out_score = torch.exp(self.t_param * torch.abs(out_score)) - torch.eye(inputs.size(0)).to(inputs.device)
        out_score = torch.mean(out_score)
        return out_score


class ReconstructImageFromFCLoss(nn.Module):
    def __init__(self, input_height, input_width, input_channel, t_param,
                 total_variance_loss_param, mutual_independence_loss_param,
                 *args, alpha_param=1, **kwargs):
        super(ReconstructImageFromFCLoss, self).__init__(*args, **kwargs)
        self.non_gaussianity_loss = NonGaussianityLoss(alpha_param=alpha_param)
        self.total_variance_loss = TotalVariationLoss(input_height, input_width, input_channel)
        self.mutual_independence_loss = MutualIndependenceLoss(t_param)
        self.mutual_independence_loss_param = mutual_independence_loss_param
        self.total_variance_loss_param = total_variance_loss_param

    def forward(self, unmixing_matrix, gradient):
        estimated_img = torch.clamp(unmixing_matrix @ gradient, min=-1., max=1.)
        non_gaussianity_score = self.non_gaussianity_loss(estimated_img)
        total_variance_score = self.total_variance_loss(estimated_img)
        mutual_independence_score = self.mutual_independence_loss(unmixing_matrix)
        out_score = (self.non_gaussianity_loss(estimated_img) +
                     self.total_variance_loss(estimated_img) * self.total_variance_loss_param +
                     self.mutual_independence_loss(unmixing_matrix) * self.mutual_independence_loss_param)
        return out_score, non_gaussianity_score, total_variance_score, mutual_independence_score
