from typing import Literal
import numpy as np
import torch
from torch import Tensor
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity, peak_signal_noise_ratio
from torchmetrics.functional.image.lpips import _NoTrainLpips, _lpips_update
import matplotlib.colors as colors


def psnr_matching(estimated_img_batch, reference_img_batch, height=32, width=32):
    pairwise_psnr = torch.empty((estimated_img_batch.shape[0], reference_img_batch.shape[0]))
    pos_estimation = torch.ones((estimated_img_batch.shape[0], reference_img_batch.shape[0])).to(bool)
    for estimated_img_idx, estimated_img in enumerate(estimated_img_batch):
        positive_estimation = torch.tensor(colors.Normalize()(np.asarray(estimated_img).reshape(3, height, width)))
        negative_estimation = torch.tensor(colors.Normalize()(np.asarray(-estimated_img).reshape(3, height, width)))
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


def psnr(estimated_img_batch, reference_img_batch, return_matches=True, height=32, width=32):
    match, is_pos, mean_psnr = psnr_matching(estimated_img_batch, reference_img_batch, height=height, width=width)
    if return_matches:
        return match, is_pos, mean_psnr
    else:
        return is_pos, mean_psnr


# WARNING: This code piece is directly taken from the torch metrics implementation
#  https://github.com/Lightning-AI/torchmetrics/blob/v1.2.1/src/torchmetrics/functional/image/lpips.py

def learned_perceptual_image_patch_similarity_wout_reduction(
        img1: Tensor,
        img2: Tensor,
        net_type: Literal["alex", "vgg", "squeeze"] = "alex",
        normalize: bool = False,
) -> Tensor:
    """The Learned Perceptual Image Patch Similarity (`LPIPS_`) calculates perceptual similarity between two images.

        LPIPS essentially computes the similarity between the activations of two image patches for some pre-defined network.
        This measure has been shown to match human perception well. A low LPIPS score means that image patches are
        perceptual similar.

        Both input image patches are expected to have shape ``(N, 3, H, W)``. The minimum size of `H, W` depends on the
        chosen backbone (see `net_type` arg).

        Args:
            img1: first set of images
            img2: second set of images
            net_type: str indicating backbone network type to use. Choose between `'alex'`, `'vgg'` or `'squeeze'`
            normalize: by default this is ``False`` meaning that the input is expected to be in the [-1,1] range. If set
                to ``True`` will instead expect input to be in the ``[0,1]`` range.
    """

    net = _NoTrainLpips(net=net_type).to(device=img1.device, dtype=img1.dtype)
    loss, total = _lpips_update(img1, img2, net, normalize)
    return loss


def lpips_matching(estimated_img_batch, reference_img_batch, verbose=False, height=32, width=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    pairwise_lpips = torch.empty((estimated_img_batch.shape[0], reference_img_batch.shape[0]))
    pos_estimation = torch.ones((estimated_img_batch.shape[0], reference_img_batch.shape[0])).to(bool)
    for estimated_img_idx, estimated_img in enumerate(estimated_img_batch):
        positive_estimation = torch.tensor(colors.Normalize()(np.asarray(estimated_img).reshape(3, height, width)))
        negative_estimation = torch.tensor(colors.Normalize()(np.asarray(-estimated_img).reshape(3, height, width)))
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

            if verbose:
                print('#########################################')
                print('estimated img idx: {}\nreference img idx:{}\npositive lpips: {}\nnegative lpips: {}'.format(
                    estimated_img_idx,
                    reference_img_idx,
                    positive_lpips,
                    negative_lpips))
                print('#########################################\n')

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
        if verbose:
            print('#########################################')
            print('pairwise lpips: {}'.format(pairwise_lpips))
            print('#########################################')

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


def lpips_matching_gpu(estimated_img_batch, reference_img_batch, height=32, width=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # finding [0, 1] normalized positive and negative images
    positive_estimation = torch.empty((estimated_img_batch.shape[0], 3, height, width))
    negative_estimation = torch.empty((estimated_img_batch.shape[0], 3, height, width))
    for estimated_img_idx, estimated_img in enumerate(estimated_img_batch):
        positive_estimation[estimated_img_idx] = torch.tensor(colors.Normalize()(
            np.asarray(estimated_img).reshape(3, height, width)))
        negative_estimation[estimated_img_idx] = torch.tensor(colors.Normalize()(
            np.asarray(-estimated_img).reshape(3, height, width)))

    # tiling reference imgs
    rep_reference_img_batch = torch.tile(reference_img_batch, (reference_img_batch.shape[0], 1, 1, 1))

    # for interleaving the estimated images
    repeat = torch.tensor([estimated_img_batch.shape[0] for _ in range(estimated_img_batch.shape[0])])

    # positive pairwise lpips
    rep_positive_estimation = torch.repeat_interleave(positive_estimation, repeat, dim=0)
    positive_lpips = learned_perceptual_image_patch_similarity_wout_reduction(
        rep_positive_estimation.to(device),
        rep_reference_img_batch.to(device),
        normalize=True,
    ).detach().to('cpu')

    # negative pairwise lpips
    rep_negative_estimation = torch.repeat_interleave(negative_estimation, repeat, dim=0)
    negative_lpips = learned_perceptual_image_patch_similarity_wout_reduction(
        rep_negative_estimation.to(device),
        rep_reference_img_batch.to(device),
        normalize=True,
    ).detach().to('cpu')

    pos_neg_lpips = torch.stack([positive_lpips, negative_lpips], dim=-1)
    lpips_min_obj = torch.min(pos_neg_lpips, 1)
    pairwise_lpips = lpips_min_obj.values
    pos_estimation = ~lpips_min_obj.indices.to(bool)
    pairwise_lpips = torch.reshape(pairwise_lpips, (estimated_img_batch.shape[0], reference_img_batch.shape[0]))
    pos_estimation = torch.reshape(pos_estimation, (estimated_img_batch.shape[0], reference_img_batch.shape[0]))

    mask = torch.zeros((estimated_img_batch.shape[0], reference_img_batch.shape[0])).to(bool)
    match = []
    is_pos = []
    mean_lpips = 0
    for _ in range(estimated_img_batch.shape[0]):
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


def lpips(estimated_img_batch, reference_img_batch, return_matches=True, verbose=False, height=32, width=32):
    match, is_pos, mean_lpips = lpips_matching(estimated_img_batch, reference_img_batch, verbose=verbose, height=height, width=width)
    if return_matches:
        return match, is_pos, mean_lpips
    else:
        return is_pos, mean_lpips


def lpips_gpu(estimated_img_batch, reference_img_batch, return_matches=True, height=32, width=32):
    match, is_pos, mean_lpips = lpips_matching_gpu(estimated_img_batch, reference_img_batch, height=height, width=width)
    if return_matches:
        return match, is_pos, mean_lpips
    else:
        return is_pos, mean_lpips
