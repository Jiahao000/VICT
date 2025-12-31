# --------------------------------------------------------
# Based on Painter, MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------

import copy
import math
import os
import random
import sys
from typing import Iterable

from PIL import Image
import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
from util.lr_sched import Scheduler
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import numpy as np
import wandb


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_prameters_from_args(model, args):
    if args.finetune_mode == 'encoder':
        for name, p in model.named_parameters():
            if name.startswith('decoder'):
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    elif args.finetune_mode == 'all':
        parameters = model.parameters()
    elif args.finetune_mode == 'encoder_no_cls_no_msk':
        for name, p in model.named_parameters():
            if name.startswith('decoder') or name == 'cls_token' or name == 'mask_token':
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    return parameters


def _reinitialize_model(base_model, base_optimizer, base_loss_scaler, clone_model, args, device):
    clone_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    clone_model.train(True)
    clone_model.to(device)
    if args.opt_type == 'sgd':
        optimizer = torch.optim.SGD(get_prameters_from_args(clone_model, args), lr=args.lr, momentum=args.opt_momentum, weight_decay=args.weight_decay)
    elif args.opt_type == 'adam':
        optimizer = torch.optim.Adam(get_prameters_from_args(clone_model, args), lr=args.lr, betas=args.opt_betas, weight_decay=args.weight_decay)
    else:
        assert args.opt_type == 'adam_w'
        optimizer = torch.optim.AdamW(get_prameters_from_args(clone_model, args), lr=args.lr, betas=args.opt_betas, weight_decay=args.weight_decay)
    scheduler = Scheduler(args.scheduler, args.steps_per_example).select_scheduler(optimizer)
    optimizer.zero_grad()
    loss_scaler = NativeScaler()
    if args.load_loss_scaler:
        loss_scaler.load_state_dict(base_loss_scaler.state_dict())
    return clone_model, optimizer, loss_scaler, scheduler


def train_on_test(base_model: torch.nn.Module,
                  clone_model: torch.nn.Module,
                  data_loader_train: Iterable,
                  data_loader_val: Iterable,
                  base_optimizer: torch.optim.Optimizer,
                  base_loss_scaler,
                  device: torch.device,
                  log_writer=None,
                  global_rank=None,
                  args=None,
                  out_dir=None,
                  iter_start: int = 0):
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    accum_iter = args.accum_iter

    # intialize the model for the current run
    model, optimizer, loss_scaler, scheduler = _reinitialize_model(base_model, base_optimizer, base_loss_scaler, clone_model, args, device)
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    dataset_len = len(data_loader_val)
    data_loader_train = iter(data_loader_train)
    data_loader_val = iter(data_loader_val)
    for data_iter_step in range(iter_start, dataset_len):
        val_data = next(data_loader_val)
        assert len(val_data) == 1
        test_samples, test_targets, test_bool_masked_pos, test_valid, test_pair, test_size = val_data[0]
        test_samples = test_samples.to(device, non_blocking=True)
        test_targets = test_targets.to(device, non_blocking=True)
        test_bool_masked_pos = test_bool_masked_pos.to(device, non_blocking=True)
        test_valid = test_valid.to(device, non_blocking=True)
        
        # test-time training
        for step_per_example in range(args.steps_per_example * accum_iter):
            train_data = next(data_loader_train)
            assert len(train_data) == 1
            samples, targets, bool_masked_pos, valid, _, _ = train_data[0]
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                # pseudo gt
                samples_gt = samples
                targets_gt = targets
                support_sample = samples[:, :, :samples.shape[2]//2, :]
                query_sample = samples[:, :, samples.shape[2]//2:, :]
                support_target = targets[:, :, :targets.shape[2]//2, :]
                query_target = targets[:, :, targets.shape[2]//2:, :]

                samples = torch.cat([support_sample, query_sample], dim=2)
                targets = torch.cat([support_target, query_target], dim=2)
                bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                latent_query = model.forward_encoder(samples, targets, 0, bool_masked_pos)
                pred_query = model.forward_decoder(latent_query)  # [N, C, H, W]
                pred_query_target = pred_query[:, :, pred_query.shape[2]//2:, :]
                new_samples = torch.cat([support_sample, query_sample], dim=2)
                new_targets = torch.cat([support_target, pred_query_target], dim=2)
                latent_support = model.forward_encoder(new_samples, new_targets, 2, bool_masked_pos)
                pred_support = model.forward_decoder(latent_support)  # [N, C, H, W]
                loss = model.forward_loss(pred_support, new_samples, new_targets, 2, bool_masked_pos, valid)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                    parameters=model.parameters(),
                                    update_grad=(step_per_example + 1) % accum_iter == 0)
            if (step_per_example + 1) % accum_iter == 0:
                if args.verbose:
                    print(f'datapoint {data_iter_step} iter {step_per_example}: lr {optimizer.param_groups[0]["lr"]} train_loss {loss_value}')
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(grad_norm=grad_norm)
            if (step_per_example + 1) % accum_iter == 0 and scheduler is not None:
                scheduler.step()

            # test
            if ((step_per_example + 1) % accum_iter == 0 and ((step_per_example + 1) // accum_iter) % args.save_freq == 0) or (step_per_example + 1 == args.steps_per_example * accum_iter):
                dst_dir = os.path.join(out_dir, str((step_per_example + 1) // accum_iter), 'results')
                if global_rank == 0:
                    os.makedirs(dst_dir, exist_ok=True)
                with torch.no_grad():
                    model.eval()
                    support_test_sample = test_samples[:, :, :test_samples.shape[2]//2, :]
                    query_test_sample = test_samples[:, :, test_samples.shape[2]//2:, :]
                    support_test_target = test_targets[:, :, :test_targets.shape[2]//2, :]
                    query_test_target = test_targets[:, :, test_targets.shape[2]//2:, :]

                    updated_test_samples = torch.cat([support_test_sample, query_test_sample], dim=2)
                    updated_test_targets = torch.cat([support_test_target, query_test_target], dim=2)
                    test_loss, y, mask = model(updated_test_samples, updated_test_targets, None, None, option=0, bool_masked_pos=test_bool_masked_pos, valid=test_valid, train_mode=False)
                    if args.verbose:
                        test_loss_value = test_loss.item()
                        print(f'datapoint {data_iter_step} iter {step_per_example}: test_loss {test_loss_value}')
                    y = model.unpatchify(y)
                    y = torch.einsum('nchw->nhwc', y).detach().cpu()
                    output = y[0, y.shape[1]//2:, :, :]
                    if "nyuv2_image2depth" in test_pair['type']:
                        room_name = test_pair['target_path'].split("/")[-2]
                        img_name = test_pair['target_path'].split("/")[-1].split(".")[0].replace('sync_depth', 'rgb')
                        out_path = dst_dir + "/" + room_name + "_" + img_name + ".png"
                        output = torch.clip((output * imagenet_std + imagenet_mean) * 10000, 0, 10000)
                        output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[test_size[1], test_size[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
                        output = output.mean(-1).int()
                        output = Image.fromarray(output.numpy())
                        output.save(out_path)
                    elif "ade20k_image2semantic" in test_pair['type'] or "coco_image2panoptic_sem_seg" in test_pair['type']:
                        img_name = os.path.basename(test_pair['target_path'])
                        out_path = os.path.join(dst_dir, img_name.replace('.jpg', '.png'))
                        output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
                        output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[test_size[1], test_size[0]], mode='bilinear').permute(0, 2, 3, 1)[0]
                        output = output.int()
                        output = Image.fromarray(output.numpy().astype(np.uint8))
                        output.save(out_path)
                    elif "derain_image2derain" in test_pair['type']:
                        rain_name = test_pair['target_path'].split("/")[-3]
                        img_name = os.path.basename(test_pair['target_path'])
                        real_dst_dir = os.path.join(dst_dir, rain_name)
                        if global_rank == 0:
                            os.makedirs(real_dst_dir, exist_ok=True)
                        out_path = os.path.join(real_dst_dir, img_name.replace('.jpg', '.png'))
                        output = output * imagenet_std + imagenet_mean
                        output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[test_size[1], test_size[0]], mode='bicubic').permute(0, 2, 3, 1)[0]
                        rgb_restored = output.numpy()
                        rgb_restored = np.clip(rgb_restored, 0, 1)
                        output = rgb_restored * 255
                        output = Image.fromarray(output.astype(np.uint8))
                        output.save(out_path)
                    elif "ssid_image2denoise" in test_pair['type']:
                        img_name = os.path.basename(test_pair['target_path'])
                        out_path = os.path.join(dst_dir, img_name.replace('.jpg', '.png'))
                        output = output * imagenet_std + imagenet_mean
                        output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[256, 256], mode='bicubic').permute(0, 2, 3, 1)[0]
                        rgb_restored = output.numpy()
                        rgb_restored = np.clip(rgb_restored, 0, 1)
                        output = rgb_restored * 255
                        output = Image.fromarray(output.astype(np.uint8))
                        output.save(out_path)
                    else:
                        assert "lol_image2enhance" in test_pair['type']
                        img_name = os.path.basename(test_pair['target_path'])
                        out_path = os.path.join(dst_dir, img_name.replace('.jpg', '.png'))
                        output = output * imagenet_std + imagenet_mean
                        output = F.interpolate(output[None, ...].permute(0, 3, 1, 2), size=[test_size[1], test_size[0]], mode='bicubic').permute(0, 2, 3, 1)[0]
                        rgb_restored = output.numpy()
                        rgb_restored = np.clip(rgb_restored, 0, 1)
                        output = rgb_restored * 255
                        output = Image.fromarray(output.astype(np.uint8))
                        output.save(out_path)
                    model.train()
        model, optimizer, loss_scaler, scheduler = _reinitialize_model(base_model, base_optimizer, base_loss_scaler, clone_model, args, device)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
