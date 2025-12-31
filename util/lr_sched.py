# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


class Scheduler(object):
    def __init__(self, name, num_epoch):
        self.name = name
        self.num_epoch = num_epoch

    def select_scheduler(self, optimizer_ft):
        scheduler = ''
        if 'multistep' == self.name:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=[15, 30], gamma=0.1)
        elif 'cosine' == self.name:
            print("This is the scheduler of CosineAnnealingLR.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=self.num_epoch, eta_min=0)
        elif 'cosinewarm' == self.name:
            print("This is the scheduler of CosineAnnealingWarmRestarts.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0=10, T_mult=1, eta_min=0)
        elif 'reducelr' == self.name:
            print('the scheduler is reduceLR.')
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=5, verbose=True)
        elif 'normal' == self.name:
            scheduler = None

        return scheduler
