"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

from . import kitti_trainer_ar, sintel_trainer_ar


def get_trainer(name):
    if name == "KITTI_AR":
        TrainFramework = kitti_trainer_ar.TrainFramework
    elif name == "SINTEL_AR":
        TrainFramework = sintel_trainer_ar.TrainFramework
    else:
        raise NotImplementedError(name)

    return TrainFramework
