import numpy as np
import torch
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity, peak_signal_noise_ratio
import matplotlib.colors as colors


def psnr_matching(estimated_img_batch, reference_img_batch):
    pairwise_psnr = torch.empty((estimated_img_batch.shape[0], reference_img_batch.shape[0]))
    pos_estimation = torch.ones((estimated_img_batch.shape[0], reference_img_batch.shape[0])).to(bool)
    for estimated_img_idx, estimated_img in enumerate(estimated_img_batch):
        positive_estimation = torch.tensor(colors.Normalize()(np.asarray(estimated_img).reshape(3, 32, 32)))
        negative_estimation = torch.tensor(colors.Normalize()(np.asarray(-estimated_img).reshape(3, 32, 32)))
        for reference_img_idx, reference_img in enumerate(reference_img_batch):
            positive_psnr = peak_signal_noise_ratio(
                torch.unsqueeze(positive_estimation, 0),
                torch.unsqueeze(reference_img, 0)
            )

            negative_psnr = peak_signal_noise_ratio(
                torch.unsqueeze(negative_estimation, 0),
                torch.unsqueeze(reference_img, 0)
            )

            if positive_psnr < negative_psnr:
                pairwise_psnr[estimated_img_idx, reference_img_idx] = positive_psnr
            else:
                pairwise_psnr[estimated_img_idx, reference_img_idx] = negative_psnr
                pos_estimation[estimated_img_idx, reference_img_idx] = False

    mask = torch.zeros((estimated_img_batch.shape[0], reference_img_batch.shape[0])).to(bool)
    match = []
    is_pos = []
    mean_psnr = 0
    for best_idx in range(estimated_img_batch.shape[0]):
        pairwise_psnr[mask] = -torch.inf
        max_psnr = torch.max(pairwise_psnr)
        mean_psnr += max_psnr.item()
        selected_match = (pairwise_psnr == max_psnr).nonzero()[0]
        match.append(selected_match)
        estimated_idx, reference_idx = selected_match[0], selected_match[1]
        is_pos.append(pos_estimation[estimated_idx, reference_idx].item())
        mask[selected_match[0], :] = True
        mask[:, selected_match[1]] = True

    return match, is_pos, (mean_psnr / estimated_img_batch.shape[0])


def psnr(estimated_img_batch, reference_img_batch, return_matches=True):
    match, is_pos, mean_psnr = psnr_matching(estimated_img_batch, reference_img_batch)
    if return_matches:
        return match, is_pos, mean_psnr
    else:
        return is_pos, mean_psnr


# TODO: we can have a faster implementation for that
def lpips_matching(estimated_img_batch, reference_img_batch):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    pairwise_lpips = torch.empty((estimated_img_batch.shape[0], reference_img_batch.shape[0]))
    pos_estimation = torch.ones((estimated_img_batch.shape[0], reference_img_batch.shape[0])).to(bool)
    for estimated_img_idx, estimated_img in enumerate(estimated_img_batch):
        positive_estimation = torch.tensor(colors.Normalize()(np.asarray(estimated_img).reshape(3, 32, 32)))
        negative_estimation = torch.tensor(colors.Normalize()(np.asarray(-estimated_img).reshape(3, 32, 32)))
        for reference_img_idx, reference_img in enumerate(reference_img_batch):
            positive_lpips = learned_perceptual_image_patch_similarity(
                torch.unsqueeze(positive_estimation, 0).to(device),
                torch.unsqueeze(reference_img, 0).to(device),
                normalize=True
            )

            negative_lpips = learned_perceptual_image_patch_similarity(
                torch.unsqueeze(negative_estimation, 0).to(device),
                torch.unsqueeze(reference_img, 0).to(device),
                normalize=True
            )

            if positive_lpips < negative_lpips:
                pairwise_lpips[estimated_img_idx, reference_img_idx] = positive_lpips
            else:
                pairwise_lpips[estimated_img_idx, reference_img_idx] = negative_lpips
                pos_estimation[estimated_img_idx, reference_img_idx] = False

    mask = torch.zeros((estimated_img_batch.shape[0], reference_img_batch.shape[0])).to(bool)
    match = []
    is_pos = []
    mean_lpips = 0
    for best_idx in range(estimated_img_batch.shape[0]):
        pairwise_lpips[mask] = torch.inf
        min_lpips = torch.min(pairwise_lpips)
        mean_lpips += min_lpips.item()
        selected_match = (pairwise_lpips == min_lpips).nonzero()[0]
        match.append(selected_match)
        estimated_idx, reference_idx = selected_match[0], selected_match[1]
        is_pos.append(pos_estimation[estimated_idx, reference_idx].item())
        mask[selected_match[0], :] = True
        mask[:, selected_match[1]] = True

    return match, is_pos, (mean_lpips / estimated_img_batch.shape[0])


def lpips(estimated_img_batch, reference_img_batch, return_matches=True):
    match, is_pos, mean_lpips = lpips_matching(estimated_img_batch, reference_img_batch)
    if return_matches:
        return match, is_pos, mean_lpips
    else:
        return is_pos, mean_lpips

