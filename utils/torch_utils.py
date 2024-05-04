"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import math
import os
import random

# import shutil

import numpy as np
import torch

# import torch.nn as nn
# import torch.nn.functional as F
from torch.optim import Optimizer
from utils.manifold_utils import MANIFOLD_BUCKET, pathmgr


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def weight_parameters(module):
    return [param for name, param in module.named_parameters() if ".weight" in name]


def bias_parameters(module):
    return [param for name, param in module.named_parameters() if ".bias" in name]


def other_parameters(module):
    return [
        param
        for name, param in module.named_parameters()
        if ".bias" not in name and ".weight" not in name
    ]


def load_checkpoint(model_path):
    # weights = torch.load(model_path)

    if "manifold" not in model_path:
        model_path = os.path.join("manifold://" + MANIFOLD_BUCKET, model_path)
    with pathmgr.open(model_path, "rb") as f:
        for i in range(3):
            try:
                weights = torch.load(f)
                break
            except Exception:
                if i == 2:
                    raise Exception

    epoch = None
    if "epoch" in weights:
        epoch = weights.pop("epoch")
    if "state_dict" in weights:
        state_dict = weights["state_dict"]
    else:
        state_dict = weights
    return epoch, state_dict


def save_checkpoint(save_path, states, file_prefixes, is_best, filename="ckpt.pth.tar"):
    def run_one_sample(save_path, state, prefix, is_best, filename):
        # torch.save(state, os.path.join(save_path, "{}_{}".format(prefix, filename)))

        if "manifold" not in save_path:
            save_path = os.path.join("manifold://" + MANIFOLD_BUCKET, save_path)
        save_path = os.path.join(save_path, "{}_{}".format(prefix, filename))
        with pathmgr.open(save_path, "wb") as f:
            for i in range(3):
                try:
                    torch.save(state, f)
                    return
                except Exception:
                    if i == 2:
                        raise Exception

    if not isinstance(file_prefixes, str):
        for (prefix, state) in zip(file_prefixes, states):
            run_one_sample(save_path, state, prefix, is_best, filename)

    else:
        run_one_sample(save_path, states, file_prefixes, is_best, filename)


def restore_model(model, pretrained_file):
    epoch, weights = load_checkpoint(pretrained_file)

    model_keys = set(model.state_dict().keys())
    weight_keys = set(weights.keys())

    # load weights by name
    weights_not_in_model = sorted(weight_keys - model_keys)
    model_not_in_weights = sorted(model_keys - weight_keys)
    if len(model_not_in_weights):
        print("Warning: There are weights in model but not in pre-trained.")
        for key in model_not_in_weights:
            print(key)
            weights[key] = model.state_dict()[key]
    if len(weights_not_in_model):
        print("Warning: There are pre-trained weights not in model.")
        for key in weights_not_in_model:
            print(key)
        from collections import OrderedDict

        new_weights = OrderedDict()
        for key in model_keys:
            new_weights[key] = weights[key]
        weights = new_weights

    model.load_state_dict(weights)
    return model


class AdamW(Optimizer):
    """Implements AdamW algorithm.

    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # according to the paper, this penalty should come after the bias correction
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group["weight_decay"] != 0:
                    p.data.add_(-group["weight_decay"], p.data)

        return loss
