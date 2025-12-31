# --------------------------------------------------------
# Based on Painter, MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------

import sys
import os
import warnings

import requests
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.append('.')
import models_painter_shared
from util.ddp_utils import DatasetTest
from util import ddp_utils


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('ADE20k semantic segmentation', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train/val set',
                        default='ADE_train_00014165')
    parser.add_argument('--split', type=str, help='prompt image split',
                        default='train', choices=['train', 'train-c', 'val'])
    parser.add_argument('--input_size', type=int, default=448)
    
    # dataset parameters
    parser.add_argument('--perturbation', type=str, help='perturbation type', 
                        default='clean')
    parser.add_argument('--level', type=str, help='severity level', 
                        default='0')
    parser.add_argument('--sample_num', default=-1, type=int,
                    help='Number of samples for test-time training (default: -1, full dataset)')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--port", default=None, type=str, help="port used to set up distributed training")
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1', args=None):
    # build model
    model = getattr(models_painter_shared, arch)()
    model.to("cuda")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_one_image(img, tgt, size, model, out_path, device, option):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    patch_size = model.module.patch_size
    _, _, h, w = tgt.shape
    num_patches = h * w // patch_size ** 2
    bool_masked_pos = torch.zeros(num_patches)
    bool_masked_pos[num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)

    valid = torch.ones_like(tgt)
    x = x.float().to(device)
    tgt = tgt.float().to(device)
    support_x = x[:, :, :x.shape[2]//2, :]
    query_x = x[:, :, x.shape[2]//2:, :]
    support_tgt = tgt[:, :, :tgt.shape[2]//2, :]
    query_tgt = tgt[:, :, tgt.shape[2]//2:, :]
    if option == 0:
        x = torch.cat([support_x, query_x], dim=2)
        tgt = torch.cat([support_tgt, query_tgt], dim=2)
    elif option == 1:
        x = torch.cat([support_x, support_tgt], dim=2)
        tgt = torch.cat([query_x, query_tgt], dim=2)
    elif option == 2:
        x = torch.cat([query_x, support_x], dim=2)
        tgt = torch.cat([query_tgt, support_tgt], dim=2)
    elif option == 3:
        x = torch.cat([support_tgt, support_x], dim=2)
        tgt = torch.cat([query_tgt, query_x], dim=2)
    elif option == 4:
        x = torch.cat([query_x, query_tgt], dim=2)
        tgt = torch.cat([support_x, support_tgt], dim=2)
    elif option == 5:
        x = torch.cat([support_tgt, query_tgt], dim=2)
        tgt = torch.cat([support_x, query_x], dim=2)
    elif option == 6:
        x = torch.cat([query_tgt, support_tgt], dim=2)
        tgt = torch.cat([query_x, support_x], dim=2)
    else:
        x = torch.cat([query_tgt, query_x], dim=2)
        tgt = torch.cat([support_tgt, support_x], dim=2)
    loss, y, mask = model(x, tgt, option=option, bool_masked_pos=bool_masked_pos.to(device), valid=valid.float().to(device))
    y = model.module.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    if option in [0, 1, 4, 5]:
        output = y[0, y.shape[1]//2:, :, :]
    else:
        output = y[0, :y.shape[1]//2, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
    output = output.int()
    output = Image.fromarray(output.numpy().astype(np.uint8))
    output.save(out_path)


if __name__ == '__main__':
    dataset_dir = "datasets/"
    args = get_args_parser()
    args = ddp_utils.init_distributed_mode(args)
    device = torch.device("cuda")

    ckpt_path = args.ckpt_path
    model = args.model
    prompt = args.prompt
    split = args.split
    input_size = args.input_size
    perturbation = args.perturbation
    level = args.level

    path_splits = ckpt_path.split('/')
    ckpt_dir, ckpt_file = path_splits[-2], path_splits[-1]
    dst_dir = os.path.join('output/inference', ckpt_dir,
                           "ade20k_semseg_inference_{}_{}_{}_size{}".format(ckpt_file, split, prompt, input_size),
                           "baseline/{}/{}/results".format(perturbation, level))

    if ddp_utils.get_rank() == 0:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        print("output_dir: {}".format(dst_dir))

    model_painter = prepare_model(ckpt_path, model, args=args)
    print('Model loaded.')

    device = torch.device("cuda")
    model_painter.to(device)

    if perturbation == 'clean':
        assert level == '0'
        img_src_dir = dataset_dir + "ade20k/images/validation"
    else:
        img_src_dir = os.path.join(dataset_dir, "ade20k-c", perturbation, level, "ade/ADEChallengeData2016/images/validation")
    dataset_val = DatasetTest(img_src_dir, input_size, ext_list=('*.jpg',), sample_num=args.sample_num)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    data_loader_val = DataLoader(dataset_val, batch_size=1, sampler=sampler_val,
                                 drop_last=False, collate_fn=ddp_utils.collate_fn, num_workers=2)

    if split == 'train':
        img2_path = dataset_dir + "ade20k/images/training/{}.jpg".format(prompt)
        tgt2_path = dataset_dir + "ade20k/annotations_with_color/training/{}.png".format(prompt)
    elif split == 'train-c':
        img2_path = os.path.join(dataset_dir, "ade20k-c", perturbation, level, "ade/ADEChallengeData2016/images/training/{}.jpg".format(prompt))
        tgt2_path = dataset_dir + "ade20k/annotations_with_color/training/{}.png".format(prompt)
    else:
        assert split == 'val'
        if perturbation == 'clean':
            img2_path = dataset_dir + "ade20k/images/validation/{}.jpg".format(prompt)
        else:
            img2_path = os.path.join(dataset_dir, "ade20k-c", perturbation, level, "ade/ADEChallengeData2016/images/validation/{}.jpg".format(prompt))
        tgt2_path = dataset_dir + "ade20k/annotations_with_color/validation/{}.png".format(prompt)

    # load the shared prompt image pair
    img2 = Image.open(img2_path).convert("RGB")
    img2 = img2.resize((input_size, input_size))
    img2 = np.array(img2) / 255.

    tgt2 = Image.open(tgt2_path)
    tgt2 = tgt2.resize((input_size, input_size))
    tgt2 = np.array(tgt2) / 255.

    model_painter.eval()
    for data in tqdm.tqdm(data_loader_val):
        """ Load an image """
        assert len(data) == 1
        img, img_path, size = data[0]
        img_name = os.path.basename(img_path)
        out_path = os.path.join(dst_dir, img_name.replace('.jpg', '.png'))

        img = np.concatenate((img2, img), axis=0)
        assert img.shape == (input_size * 2, input_size, 3)
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)

        assert tgt.shape == (input_size * 2, input_size, 3)
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        # make random mask reproducible (comment out to make it change)
        torch.manual_seed(2)
        run_one_image(img, tgt, size, model_painter, out_path, device, 0)
