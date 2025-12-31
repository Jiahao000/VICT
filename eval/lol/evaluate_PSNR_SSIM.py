import sys
import os

import argparse

import torch
import numpy as np
import glob
import tqdm

from PIL import Image

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss


def get_args_parser():
    parser = argparse.ArgumentParser('low-light enhancement', add_help=False)
    parser.add_argument('--pred_path', type=str, 
                        help='path to the prediction results in png', required=True)
    parser.add_argument('--gt_path', type=str, default='datasets/light_enhance/eval15/high', 
                        help='root path to the groundtruth data', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()

    pred_path = args.pred_path
    gt_path = args.gt_path

    img_path_list = glob.glob(os.path.join(pred_path, "results", "*.png"))
    img_path_list = sorted(img_path_list)

    psnr_val_rgb = []
    ssim_val_rgb = []
    for img_path in tqdm.tqdm(img_path_list):
        """ Load an image """
        img_name = os.path.basename(img_path)
        rgb_restored = Image.open(img_path).convert("RGB")
        rgb_restored = np.array(rgb_restored) / 255.

        # load gt
        rgb_gt = Image.open(os.path.join(gt_path, img_name)).convert("RGB")
        rgb_gt = np.array(rgb_gt) / 255.

        psnr = psnr_loss(rgb_restored, rgb_gt)
        ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True)
        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)
        print("PSNR:", psnr, ", SSIM:", ssim, img_name, rgb_restored.shape)

        with open(os.path.join(pred_path, 'psnr_ssim.txt'), 'a') as f:
            f.write(img_name+' ---->'+" PSNR: %.4f, SSIM: %.4f] " % (psnr, ssim)+'\n')

    psnr_val_rgb = sum(psnr_val_rgb) / len(img_path_list)
    ssim_val_rgb = sum(ssim_val_rgb) / len(img_path_list)
    print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
    with open(os.path.join(pred_path, 'psnr_ssim.txt'), 'a') as f:
        f.write("PSNR: %.4f, SSIM: %.4f] " % (psnr_val_rgb, ssim_val_rgb)+'\n')
