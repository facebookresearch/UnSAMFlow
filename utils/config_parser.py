"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import json
import os

from easydict import EasyDict
from utils.manifold_utils import pathmgr


def update_config(base_cfg, new_cfg):
    for key in new_cfg:
        if key not in base_cfg:
            base_cfg[key] = new_cfg[key]
        elif type(base_cfg[key]) == EasyDict and type(new_cfg[key]) == EasyDict:
            update_config(base_cfg[key], new_cfg[key])
        else:
            base_cfg[key] = new_cfg[key]

    return base_cfg


def init_config(cfg_file):
    with open(pathmgr.get_local_path(cfg_file)) as f:
        cfg = EasyDict(json.load(f))

    if "base_configs" in cfg:
        base_cfg_file = os.path.join(os.path.dirname(cfg_file), cfg.base_configs)
        with open(pathmgr.get_local_path(base_cfg_file)) as f:
            base_cfg = EasyDict(json.load(f))
        cfg = update_config(base_cfg, cfg)

    return cfg
