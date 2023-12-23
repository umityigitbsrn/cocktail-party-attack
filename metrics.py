import numpy as np
import torch


def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    psnr_arr = []
    if img_batch.ndim != 4:
        img_batch = torch.unsqueeze(img_batch, 0)
        ref_batch = torch.unsqueeze(ref_batch, 0)
    for img_in, img_ref in zip(img_batch, ref_batch):
        mse = ((img_in - img_ref) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            psnr_arr.append(10 * torch.log10(factor ** 2 / mse))
    return np.asarray(psnr_arr).mean()


def psnr_matching(img_batch, ref_batch):
    pairwise_psnr = torch.zeros((img_batch.shape[0], ref_batch.shape[0]))
    for row_idx, img_in in enumerate(img_batch):
        for col_idx, img_ref in enumerate(ref_batch):
            score = psnr(img_in, img_ref)
            pairwise_psnr[row_idx, col_idx] = score.item()

    mask = torch.zeros((img_batch.shape[0], ref_batch.shape[0])).to(bool)
    match = []
    for best_idx in range(img_batch.shape[0]):
        pairwise_psnr[mask] = -torch.inf
        torch.argmax(pairwise_psnr)
        selected_match = (pairwise_psnr == torch.max(pairwise_psnr)).nonzero()[0]
        match.append(selected_match)
        mask[selected_match[0], :] = True
        mask[:, selected_match[1]] = True

    return match
