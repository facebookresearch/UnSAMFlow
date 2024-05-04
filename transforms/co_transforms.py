"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import numbers
import random

import numpy as np


def get_co_transforms(aug_args):
    transforms = []
    if aug_args.swap:  # swap first and second frame
        transforms.append(RandomTemporalSwap())
    if aug_args.hflip:
        transforms.append(RandomHorizontalFlip())
    if aug_args.crop:
        transforms.append(RandomCrop(aug_args.para_crop))
    return Compose(transforms)


class Compose:
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, imgs, full_segs, key_objs, target):
        for t in self.co_transforms:
            imgs, full_segs, key_objs, target = t(imgs, full_segs, key_objs, target)
        return imgs, full_segs, key_objs, target


class RandomCrop:
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs, full_segs, key_objs, target):
        h, w, _ = imgs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return imgs, target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        imgs = [img[y1 : y1 + th, x1 : x1 + tw] for img in imgs]
        full_segs = [full_seg[y1 : y1 + th, x1 : x1 + tw] for full_seg in full_segs]
        key_objs = [key_obj[y1 : y1 + th, x1 : x1 + tw] for key_obj in key_objs]

        if target != {}:
            raise NotImplementedError(
                "RandomCrop currently does not take ground-truth labels"
            )

        return imgs, full_segs, key_objs, target


class RandomTemporalSwap:
    """Randomly swap first and second frames"""

    def __call__(self, imgs, full_segs, key_objs, target):

        if random.random() < 0.5:
            imgs = imgs[::-1]
            full_segs = full_segs[::-1]
            key_objs = key_objs[::-1]

            if target != {}:
                raise NotImplementedError(
                    "RandomTemporalSwap currently does not take ground-truth labels"
                )

        return imgs, full_segs, key_objs, target


class RandomHorizontalFlip:
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5"""

    def __call__(self, imgs, full_segs, key_objs, target):
        if random.random() < 0.5:
            imgs = [np.copy(np.fliplr(im)) for im in imgs]
            full_segs = [np.copy(np.fliplr(full_seg)) for full_seg in full_segs]
            key_objs = [np.copy(np.fliplr(key_obj)) for key_obj in key_objs]

            if target != {}:
                raise NotImplementedError(
                    "RandomHorizontalFlip currently does not take ground-truth labels"
                )
        return imgs, full_segs, key_objs, target
