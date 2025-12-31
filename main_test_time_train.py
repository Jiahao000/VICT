# --------------------------------------------------------
# Based on Painter, MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import random
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"  # version check

import util.lr_decay as lrd
import util.misc as misc
from util import ddp_utils
from util.misc import get_parameter_groups
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed

import models_painter_shared_tt
from engine_test_time import train_on_test

from data.pairdataset_tt import PairDatasetTT
import data.pair_transforms as pair_transforms
from util.masking_generator import MaskingGenerator
from data.sampler import DistributedSamplerWrapper

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


def get_args_parser():
    parser = argparse.ArgumentParser('VICT', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--steps_per_example', default=1, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Prompt parameters
    parser.add_argument('--prompt_img_path', type=str, default='ADE_train_00014165',
                        help='prompt image path, usually in train set')
    parser.add_argument('--prompt_tgt_path', type=str, default='ADE_train_00014165',
                        help='prompt target path, usually in train set')
    parser.add_argument('--loss_func', type=str, default='smoothl1', choices=['l1', 'l2', 'l1l2', 'smoothl1'],
                    help='loss function')
    
    # Model parameters
    parser.add_argument('--model', default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int, nargs='+',
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--num_mask_patches', default=784, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=392)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)
    parser.add_argument('--stop_grad_patch_embed', action='store_true',
                        help='stop-grad after first conv, or patch embedding')
    parser.set_defaults(stop_grad_patch_embed=False)
    parser.add_argument('--finetune', type=str, default='', required=True,
                        help='finetune from checkpoint')
    parser.add_argument('--finetune_mode', type=str, default='all', 
                        help='all, encoder, encoder_no_cls_no_msk.')
    parser.add_argument('--min_random_scale', default=0.3, type=float,
                        help='Minimal random scale for randomresizecrop (default: 0.3)')
    parser.add_argument('--last_norm_instance', action='store_true', default=False,
                    help='use instance norm to normalize each channel map before the decoder layer')
    parser.add_argument('--half_mask_ratio', default=1.0, type=float,
                        help='ratio of using half mask during training (default: 1.0)')
    parser.add_argument('--use_checkpoint', action='store_true', default=False,
                    help='use checkpoint to save GPU memory')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay (default: 0.)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save checkpoints frequency')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--opt_type', default='adam_w', help='adam, adam_w, sgd')
    parser.add_argument('--opt_momentum', default=0.9, type=float, help='Optimizer momentum')
    parser.add_argument("--scheduler", type=str, default='normal', choices=['normal', 'multistep', 'cosine', 'cosinewarm', 'reducelr'],
                        help="scheduler for training")
    parser.add_argument('--load_loss_scaler', action='store_true')
    parser.set_defaults(load_loss_scaler=False)

    # Dataset parameters
    parser.add_argument('--sample_num', default=-1, type=int,
                    help='Number of samples for test-time training (default: -1, full dataset)')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--val_json_path', default='./', nargs='+',type=str,
                        help='json path')
    parser.add_argument('--single_crop', action='store_true', default=False, 
                        help='single_crop training')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to save tensorboard log')
    parser.add_argument('--infer_dir', default='./output_dir',
                        help='path where to save inference results')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--use_two_pairs', action='store_true',
                        help='concatenate two pairs of images')
    parser.set_defaults(use_two_pairs=True)
    parser.add_argument('--load_loss_scalar', action='store_true')
    parser.set_defaults(load_loss_scalar=False)
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(verbose=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--port', default=None, type=str, 
                        help='port used to set up distributed training')
	# misc
    parser.add_argument('--log_wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

    return parser.parse_args()


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # define the base model
    model = models_painter_shared_tt.__dict__[args.model](loss_func=args.loss_func)
    # define the cloned model
    clone_model = models_painter_shared_tt.__dict__[args.model](loss_func=args.loss_func)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained model from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        rm_key_list = ['decoder_embed.weight', 'decoder_embed.bias',  'mask_token']
        if args.last_norm_instance:
            rm_key_list.extend(['norm.weight', 'norm.bias'])
        for k in rm_key_list:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate patch embedding
        if "patch32" in args.model:
            patch_weight = checkpoint['model']['patch_embed.proj.weight']
            new_patch_weight = torch.nn.functional.interpolate(patch_weight, size=(32, 32), mode='bicubic', align_corners=False)
            checkpoint['model']['patch_embed.proj.weight'] = new_patch_weight

        # interpolate position embedding
        if "painter" not in args.model:
            interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size[0] // patch_size, args.input_size[1] // patch_size)
    args.patch_size = patch_size

    # simple augmentation
    if not args.single_crop:
        transform_train = pair_transforms.Compose([
                pair_transforms.RandomResizedCrop(args.input_size[1], scale=(args.min_random_scale, 1.0), interpolation=3),  # 3 is bicubic
                pair_transforms.RandomHorizontalFlip(),
                pair_transforms.ToTensor(),
                pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_train2 = pair_transforms.Compose([
                pair_transforms.RandomResizedCrop(args.input_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
                pair_transforms.ToTensor(),
                pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_train3 = pair_transforms.Compose([
                pair_transforms.RandomResizedCrop(args.input_size[1], scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
                pair_transforms.ToTensor(),
                pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_train_seccrop = pair_transforms.Compose([
                pair_transforms.RandomResizedCrop(args.input_size, scale=(args.min_random_scale, 1.0), ratio=(0.3, 0.7), interpolation=3),  # 3 is bicubic
                ])
    else:
        transform_train = pair_transforms.Compose([
                pair_transforms.Resize((args.input_size[1], args.input_size[1]), interpolation=3),  # 3 is bicubic
                pair_transforms.ToTensor(),
                pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_train2 = None
        transform_train3 = None
        transform_train_seccrop = None
    transform_val = pair_transforms.Compose([
            pair_transforms.Resize((args.input_size[1], args.input_size[1]), interpolation=3),  # 3 is bicubic
            pair_transforms.ToTensor(),
            pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    masked_position_generator = MaskingGenerator(
        args.window_size, num_masking_patches=args.num_mask_patches,
        max_num_patches=args.max_mask_patches_per_block,
        min_num_patches=args.min_mask_patches_per_block,
    )
    dataset_train = PairDatasetTT(
        args.data_path, 
        args.val_json_path, 
        args.prompt_img_path, 
        args.prompt_tgt_path, 
        batch_size=args.batch_size, 
        steps_per_example=args.steps_per_example * args.accum_iter, 
        transform=transform_train, 
        transform2=transform_train2, 
        transform3=transform_train3, 
        transform_seccrop=transform_train_seccrop, 
        masked_position_generator=masked_position_generator, 
        single_crop=args.single_crop, 
        use_two_pairs=args.use_two_pairs, 
        half_mask_ratio=1.0, 
        start_index=0,
        sample_num=args.sample_num
    )
    dataset_val = PairDatasetTT(
        args.data_path, 
        args.val_json_path, 
        args.prompt_img_path, 
        args.prompt_tgt_path, 
        batch_size=1, 
        transform=transform_val, 
        masked_position_generator=masked_position_generator, 
        single_crop=args.single_crop, 
        use_two_pairs=args.use_two_pairs, 
        half_mask_ratio=1.0, 
        start_index=0,
        sample_num=args.sample_num
    )
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        print("number of samples (train): %d" % len(dataset_train))
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.log_wandb:
        experiment = args.log_dir.split('/')[-2]
        if args.resume == '':
            wandb.init(project="VICT", name=experiment, config=args)
        else:
            wandb.init(project="VICT", name=experiment, config=args, resume=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=1,
        collate_fn=ddp_utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        collate_fn=ddp_utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        )
    
    model.to(device)
    clone_model.to(device)
    model_without_ddp = model
    clone_model_without_ddp = clone_model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    print("loss function: %s" % args.loss_func)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        clone_model = torch.nn.parallel.DistributedDataParallel(clone_model, device_ids=[args.gpu])

        model_without_ddp = model.module
        clone_model_without_ddp = clone_model.module

    optimizer = None
    if args.load_loss_scaler:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None

    start_time = time.time()
    test_stats = train_on_test(
        model_without_ddp,
        clone_model_without_ddp,
        data_loader_train, data_loader_val,
        optimizer, loss_scaler, device,
        log_writer=log_writer,
        global_rank=global_rank,
        args=args,
        out_dir=args.infer_dir,
        iter_start=0
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if global_rank == 0 and args.log_wandb:
        wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
