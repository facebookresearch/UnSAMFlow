"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import datetime

curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

import argparse
import os
import pprint

import torch

from utils.config_parser import init_config

# from utils.logger import init_logger

torch.backends.cudnn.benchmark = True
import numpy as np

import pkg_resources

from datasets.get_dataset import get_dataset

from fblearner.flow.util.visualization_utils import summary_writer

from losses.get_loss import get_loss

from models.get_model import get_model

from trainer.get_trainer import get_trainer

# our internal file system; please comment out this line and change I/O to your own file system
from utils.manifold_utils import MANIFOLD_BUCKET, MANIFOLD_PATH, pathmgr

from utils.torch_utils import init_seed


def main_ddp(rank, world_size, cfg):
    init_seed(cfg.seed)

    # set up distributed process groups
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )

    device = torch.device("cuda:%d" % rank)
    torch.cuda.set_device(device)
    print(f"Use GPU {rank} ({torch.cuda.get_device_name(rank)}) for training")

    # prepare data
    train_sets, valid_sets, train_sets_epoches = get_dataset(cfg.data)
    if rank == 0:
        print(
            "train sets: "
            + ", ".join(
                ["{} ({} samples)".format(ds.name, len(ds)) for ds in train_sets]
            )
        )
        print(
            "val sets: "
            + ", ".join(
                ["{} ({} samples)".format(ds.name, len(ds)) for ds in valid_sets]
            )
        )

    train_sets_epoches = [np.inf if e == -1 else e for e in train_sets_epoches]

    train_loaders, valid_loaders = [], []
    for ds in train_sets:
        sampler = torch.utils.data.DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        train_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.train.batch_size // world_size,
            num_workers=cfg.train.workers // world_size,
            pin_memory=True,
            sampler=sampler,
        )
        train_loaders.append(train_loader)

    if rank == 0:
        # prepare tensorboard
        writer = summary_writer(log_dir=cfg.save_root)

        # prepare validation dataset
        for ds in valid_sets:
            valid_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=4,
                num_workers=4,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
            )
            valid_loaders.append(valid_loader)
        valid_size = sum([len(loader) for loader in valid_loaders])
        if cfg.train.valid_size == 0:
            cfg.train.valid_size = valid_size
        cfg.train.valid_size = min(cfg.train.valid_size, valid_size)

    else:
        writer = None
        valid_loaders = []

    # prepare model
    model = get_model(cfg.model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank,
    )

    # prepare loss
    loss = get_loss(cfg.loss)

    # prepare training scipt
    trainer = get_trainer(cfg.trainer)(
        train_loaders,
        valid_loaders,
        model,
        loss,
        cfg.save_root,
        cfg.train,
        resume=cfg.resume,
        train_sets_epoches=train_sets_epoches,
        summary_writer=writer,
        rank=rank,
        world_size=world_size,
    )

    trainer.train()

    torch.distributed.destroy_process_group()


def main(args, run_id=None):

    # resuming
    if args.resume is not None:
        args.config = os.path.join(
            "manifold://", MANIFOLD_BUCKET, MANIFOLD_PATH, args.resume, "config.json"
        )
    else:
        args.config = pkg_resources.resource_filename(__name__, args.config)

    # load config
    cfg = init_config(args.config)
    cfg.train.n_gpu = args.n_gpu

    # DEBUG options
    cfg.train.DEBUG = args.DEBUG
    if args.DEBUG:
        cfg.data.update(
            {
                "epoches_raw": 3,
            }
        )
        cfg.train.update(
            {
                "batch_size": 4,
                "epoch_num": 5,
                "epoch_size": 20,
                "print_freq": 1,
                "record_freq": 1,
                "val_epoch_size": 2,
                "valid_size": 4,
                "save_iter": 2,
            }
        )
        if "stage1" in cfg.train:
            cfg.train.stage1.update({"epoch": 5})
        if "stage2" in cfg.train:
            cfg.train.stage2.update({"epoch": 5})

    # pretrained model
    if args.model is not None:
        cfg.train.pretrained_model = args.model

    # init save_root: store files by curr_time
    if args.resume is not None:
        cfg.resume = True
        cfg.save_root = os.path.join(
            "manifold://", MANIFOLD_BUCKET, MANIFOLD_PATH, args.resume
        )
    else:
        cfg.resume = False
        args.name = os.path.basename(args.config)[:-5]

        dirname = curr_time + "_" + args.name
        if run_id is not None:
            dirname = dirname + "_f" + str(run_id)
        if args.DEBUG:
            dirname = "_DEBUG_" + dirname

        cfg.save_root = os.path.join(
            "manifold://",
            MANIFOLD_BUCKET,
            MANIFOLD_PATH,
            args.exp_folder,
            dirname,
        )

        ## for the manifold file system

        if not pathmgr.exists(cfg.save_root):
            pathmgr.mkdirs(cfg.save_root)

            pathmgr.copy_from_local(
                args.config, os.path.join(cfg.save_root, "config.json")
            )

            if "base_configs" in cfg:
                pathmgr.copy_from_local(
                    os.path.join(os.path.dirname(args.config), cfg.base_configs),
                    os.path.join(cfg.save_root, cfg.base_configs),
                )

        """
        ## for the linux file system
        os.makedirs(cfg.save_root)
        os.system(
            "cp {} {}".format(args.config, os.path.join(cfg.save_root, "config.json"))
        )
        if "base_configs" in cfg:
            os.system(
                "cp {} {}".format(
                    os.path.join(os.path.dirname(args.config), cfg.base_configs),
                    os.path.join(cfg.save_root, cfg.base_configs),
                )
            )
        """

    print("=> will save everything to {}".format(cfg.save_root))

    # show configurations
    cfg_str = pprint.pformat(cfg)
    print("=> configurations \n " + cfg_str)

    # spawn ddp
    world_size = args.n_gpu
    torch.multiprocessing.spawn(
        main_ddp,
        args=(world_size, cfg),
        nprocs=world_size,
    )

    print("Completed!")
    return


def invoke_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/base_kitti.json")
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("--exp_folder", default="other")
    parser.add_argument("-n", "--name", default=None)
    parser.add_argument("-r", "--resume", default=None)
    parser.add_argument("--n_gpu", type=int, default=2)
    parser.add_argument("--DEBUG", action="store_true")
    args = parser.parse_args()

    main(args)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
