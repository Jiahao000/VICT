# --------------------------------------------------------
# Based on Painter, MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------

import sys
import os
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('.')
import models_painter_shared
from util import ddp_utils


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


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

    bool_masked_pos = torch.zeros(model.module.patch_embed.num_patches)
    bool_masked_pos[model.module.patch_embed.num_patches//2:] = 1
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
    output = torch.clip((output * imagenet_std + imagenet_mean) * 10000, 0, 10000)
    output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
    output = output.mean(-1).int()
    output = Image.fromarray(output.numpy())
    output.save(out_path)
    

def get_args_parser():
    parser = argparse.ArgumentParser('NYU Depth V2', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train/val set',
                        default='study_room_0005b/rgb_00094')
    parser.add_argument('--split', type=str, help='prompt image split',
                        default='train', choices=['train', 'train-c', 'val'])
    parser.add_argument('--input_size', type=int, default=448)

    # dataset parameters
    parser.add_argument('--perturbation', type=str, help='perturbation type', 
                        default='clean')
    parser.add_argument('--level', type=str, help='severity level', 
                        default='0')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--port", default=None, type=str, help="port used to set up distributed training")
    return parser.parse_args()


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
                           "nyuv2_depth_inference_{}_{}_{}_size{}".format(ckpt_file, split, prompt, input_size),
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
        img_src_dir = dataset_dir + "nyu_depth_v2/official_splits/test"
    else:
        img_src_dir = os.path.join(dataset_dir, "nyu_depth_v2-c", perturbation, level, "official_splits/test")
    img_path_list = glob.glob(img_src_dir + "/*/rgb*g")
    if split == 'train':
        img2_path = dataset_dir + "nyu_depth_v2/sync/{}.jpg".format(prompt)
        tgt_path = dataset_dir + "nyu_depth_v2/sync/{}.png".format(prompt.replace('rgb', 'sync_depth'))
    elif split == 'train-c':
        img2_path = os.path.join(dataset_dir, "nyu_depth_v2-c", perturbation, level, "sync/{}.jpg".format(prompt))
        tgt_path = dataset_dir + "nyu_depth_v2/sync/{}.png".format(prompt.replace('rgb', 'sync_depth'))
    else:
        assert split == 'val'
        if perturbation == 'clean':
            img2_path = dataset_dir + "nyu_depth_v2/official_splits/test/{}.jpg".format(prompt)
        else:
            img2_path = os.path.join(dataset_dir, "nyu_depth_v2-c", perturbation, level, "official_splits/test/{}.jpg".format(prompt))
        tgt_path = dataset_dir + "nyu_depth_v2/official_splits/test/{}.png".format(prompt.replace('rgb', 'sync_depth'))
    tgt2_path = tgt_path

    res, hres = input_size, input_size

    model_painter.eval()
    for img_path in tqdm.tqdm(img_path_list):
        room_name = img_path.split("/")[-2]
        img_name = img_path.split("/")[-1].split(".")[0]
        out_path = dst_dir + "/" + room_name + "_" + img_name + ".png"
        img = Image.open(img_path).convert("RGB")
        size = img.size
        img = img.resize((res, hres))
        img = np.array(img) / 255.
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.
        img = np.concatenate((img2, img), axis=0)
        assert img.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        tgt = Image.open(tgt_path)
        tgt = np.array(tgt) / 10000.
        tgt = tgt * 255
        tgt = Image.fromarray(tgt).convert("RGB")
        tgt = tgt.resize((res, hres))
        tgt = np.array(tgt) / 255.
        tgt2 = Image.open(tgt2_path)
        tgt2 = np.array(tgt2) / 10000.
        tgt2 = tgt2 * 255
        tgt2 = Image.fromarray(tgt2).convert("RGB")
        tgt2 = tgt2.resize((res, hres))
        tgt2 = np.array(tgt2) / 255.
        tgt = np.concatenate((tgt2, tgt), axis=0)

        assert tgt.shape == (2 * res, res, 3)
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        torch.manual_seed(2)
        run_one_image(img, tgt, size, model_painter, out_path, device, 0)
