"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import cv2
import numpy as np

import torch
from torch.nn import functional as F


class ArrayToTensor:
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, all_data):
        imgs, full_segs, key_objs = all_data
        imgs = [torch.from_numpy(img.transpose((2, 0, 1))).float() for img in imgs]
        full_segs = [
            torch.from_numpy(full_seg.transpose((2, 0, 1))).float()
            for full_seg in full_segs
        ]

        if key_objs is not None:
            key_objs = [
                torch.from_numpy(key_obj.transpose((2, 0, 1))).float()
                for key_obj in key_objs
            ]

        return imgs, full_segs, key_objs


class Zoom:
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, all_data):
        imgs, full_segs, key_objs = all_data
        imgs = [cv2.resize(img, (self.new_w, self.new_h)) for img in imgs]
        full_segs = [
            cv2.resize(
                full_seg, (self.new_w, self.new_h), interpolation=cv2.INTER_NEAREST
            )[:, :, None]
            for full_seg in full_segs
        ]

        if key_objs is not None:
            new_key_objs = []
            for key_obj in key_objs:
                if key_obj.shape[-1] == 0:  ## no key obj found
                    new_key_obj = np.zeros((self.new_h, self.new_w, 0), dtype=np.uint8)
                elif key_obj.shape[-1] == 1:
                    new_key_obj = cv2.resize(
                        key_obj,
                        (self.new_w, self.new_h),
                        interpolation=cv2.INTER_NEAREST,
                    )[:, :, None]
                else:
                    new_key_obj = cv2.resize(
                        key_obj,
                        (self.new_w, self.new_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                new_key_objs.append(new_key_obj)
        else:
            new_key_objs = None

        return imgs, full_segs, new_key_objs


def full_segs_to_adj_maps(full_segs, win_size=9, pad_mode="replicate"):
    """
    Input: full_segs: [B, 1, H, W]
    Output: adj_maps: [B, win_size * win_size, H, W]
    """

    r = (win_size - 1) // 2
    b, _, h, w = full_segs.shape
    full_segs_padded = F.pad(full_segs, (r, r, r, r), mode=pad_mode)

    nb = F.unfold(full_segs_padded, [win_size, win_size])
    nb = nb.reshape((b, win_size * win_size, h, w))

    adj_maps = (full_segs == nb).float()
    return adj_maps
