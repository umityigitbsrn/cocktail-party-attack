import numpy as np
import torch
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity, peak_signal_noise_ratio
import matplotlib.colors as colors


def psnr_matching(estimated_img_batch, reference_img_batch):
    normalized_imgs = []
    for img in estimated_img_batch:
        normalized_imgs.append(torch.tensor(colors.Normalize()(np.asarray(img).reshape(3, 32, 32))))

    normalized_imgs = torch.stack(normalized_imgs, dim=0)

    pairwise_psnr = torch.empty((normalized_imgs.shape[0], reference_img_batch.shape[0]))
    for estimated_idx, estimated_img in enumerate(normalized_imgs):
        for reference_img_idx, reference_img in enumerate(reference_img_batch):
            pairwise_psnr[estimated_idx, reference_img_idx] = peak_signal_noise_ratio(
                torch.unsqueeze(estimated_img, 0), torch.unsqueeze(reference_img, 0))

    mask = torch.zeros((normalized_imgs.shape[0], reference_img_batch.shape[0])).to(bool)
    match = []
    mean_psnr = 0
    for best_idx in range(normalized_imgs.shape[0]):
        pairwise_psnr[mask] = -torch.inf
        max_lpips = torch.max(pairwise_psnr)
        mean_psnr += max_lpips.item()
        selected_match = (pairwise_psnr == max_lpips).nonzero()[0]
        match.append(selected_match)
        mask[selected_match[0], :] = True
        mask[:, selected_match[1]] = True

    return match, (mean_psnr / normalized_imgs.shape[0])


def psnr(estimated_img_batch, reference_img_batch, return_matches=True):
    match_positive, psnr_positive = psnr_matching(estimated_img_batch, reference_img_batch)
    match_negative, psnr_negative = psnr_matching(-estimated_img_batch, reference_img_batch)
    if psnr_positive > psnr_negative:
        if return_matches:
            return match_positive, psnr_positive, True
        else:
            return psnr_positive, True
    else:
        if return_matches:
            return match_negative, psnr_negative, False
        else:
            return psnr_negative, False


# TODO: we can have a faster implementation for that
def lpips_matching(estimated_img_batch, reference_img_batch):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    normalized_imgs = []
    for img in estimated_img_batch:
        normalized_imgs.append(torch.tensor(colors.Normalize()(np.asarray(img).reshape(3, 32, 32))))

    normalized_imgs = torch.stack(normalized_imgs, dim=0)

    pairwise_lpips = torch.empty((normalized_imgs.shape[0], reference_img_batch.shape[0]))
    for estimated_idx, estimated_img in enumerate(normalized_imgs):
        for reference_img_idx, reference_img in enumerate(reference_img_batch):
            pairwise_lpips[estimated_idx, reference_img_idx] = learned_perceptual_image_patch_similarity(
                torch.unsqueeze(estimated_img, 0).to(device), torch.unsqueeze(reference_img, 0).to(device),
                normalize=True)

    mask = torch.zeros((normalized_imgs.shape[0], reference_img_batch.shape[0])).to(bool)
    match = []
    mean_lpips = 0
    for best_idx in range(normalized_imgs.shape[0]):
        pairwise_lpips[mask] = torch.inf
        min_lpips = torch.min(pairwise_lpips)
        mean_lpips += min_lpips.item()
        selected_match = (pairwise_lpips == min_lpips).nonzero()[0]
        match.append(selected_match)
        mask[selected_match[0], :] = True
        mask[:, selected_match[1]] = True

    return match, (mean_lpips / normalized_imgs.shape[0])


def lpips(estimated_img_batch, reference_img_batch, return_matches=True):
    match_positive, lpips_positive = lpips_matching(estimated_img_batch, reference_img_batch)
    match_negative, lpips_negative = lpips_matching(-estimated_img_batch, reference_img_batch)
    if lpips_positive < lpips_negative:
        if return_matches:
            return match_positive, lpips_positive, True
        else:
            return lpips_positive, True
    else:
        if return_matches:
            return match_negative, lpips_negative, False
        else:
            return lpips_negative, False

