import argparse
import scipy.io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_filter
import numpy as np
import os
from PIL import Image


def compute_ssim(img1, img2):
    sigma = 1.5
    filter_size = 11
    K = [0.01, 0.03]
    L = 1.

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    # Convert img1 and img2 to double
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    # Compute local means using the window

    mu1 = gaussian_filter(img1, sigma=sigma, mode='nearest', truncate=filter_size)
    mu2 = gaussian_filter(img2, sigma=sigma, mode='nearest', truncate=filter_size)

    # Compute local variances and covariances
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 ** 2, sigma=sigma, mode='nearest', truncate=filter_size) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=sigma, mode='nearest', truncate=filter_size) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma, mode='nearest', truncate=filter_size) - mu1_mu2

    # Calculate SSIM map
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    mssim = np.mean(ssim_map)
    return mssim


def get_args_parser():
    parser = argparse.ArgumentParser('Denoising', add_help=False)
    parser.add_argument('--pred_path', type=str, 
                        help='path to the prediction results in png', required=True)
    parser.add_argument('--gt_path', type=str, default='datasets/denoise/val/groundtruth', 
                        help='root path to the groundtruth data', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()

    total_psnr = 0
    total_ssim = 0
    for i in range(1, 41):
        for k in range(1, 33):
            denoised_path = os.path.join(args.pred_path, '%04d_%02d.png' % (i, k))
            denoised_patch = Image.open(denoised_path)
            denoised_patch = np.array(denoised_patch) / 255.

            gt_path = os.path.join(args.gt_path, '%04d_%02d.png' % (i, k))
            gt_patch = Image.open(gt_path)
            gt_patch = np.array(gt_patch) / 255.

            ssim_val = compute_ssim(denoised_patch, gt_patch)
            psnr_val = psnr(denoised_patch, gt_patch)

            total_ssim += ssim_val
            total_psnr += psnr_val

    qm_psnr = total_psnr / (40 * 32)
    qm_ssim = total_ssim / (40 * 32)

    print(f'PSNR: {qm_psnr} SSIM: {qm_ssim}')
