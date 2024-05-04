"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import argparse

import os

import torch
from datasets.flow_datasets import KITTIFlowEval, Sintel

from models.get_model import get_model

from torchvision import transforms
from tqdm import tqdm
from transforms import input_transforms
from utils.config_parser import init_config
from utils.flow_utils import resize_flow, writeFlowKITTI, writeFlowSintel
from utils.manifold_utils import MANIFOLD_BUCKET, MANIFOLD_PATH, pathmgr
from utils.torch_utils import restore_model

parser = argparse.ArgumentParser(
    description="create_submission",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--model-folder",
    required=True,
    type=str,
    help="the model folder (that contains the configuration file)",
)
parser.add_argument(
    "--output-dir",
    default=None,
    type=str,
    help="Output directory; default is test_flow under the folder of the model",
)
parser.add_argument(
    "--trained-model",
    required=True,
    default="model_ckpt.pth.tar",
    type=str,
    help="trained model path in the model folder",
)
parser.add_argument(
    "--dataset", type=str, choices=["sintel", "kitti"], help="sintel/kitti"
)
parser.add_argument(
    "--subset", type=str, default="test", choices=["train", "test"], help="train/test"
)


def tensor2array(tensor):
    return tensor.detach().cpu().numpy().transpose([0, 2, 3, 1])


@torch.no_grad()
def create_sintel_submission(model, args):
    """Create submission for the Sintel leaderboard"""

    input_transform = transforms.Compose(
        [
            input_transforms.Zoom(args.img_height, args.img_width),
            input_transforms.ArrayToTensor(),
        ]
    )

    # start inference
    model.eval()
    for dstype in ["final", "clean"]:
        # ds_dir = os.path.join(args.output_dir, dstype)
        ds_dir_local = os.path.join(args.output_local_dir, dstype)
        ds_dir_bw_local = os.path.join(args.output_local_dir + "_bw", dstype)
        # pathmgr.mkdirs(ds_dir)
        os.makedirs(ds_dir_local, exist_ok=True)
        os.makedirs(ds_dir_bw_local, exist_ok=True)

        dataset = Sintel(
            args.root_sintel,
            args.full_seg_root_sintel,
            None,
            name="sintel-" + dstype,
            dataset_type=dstype,
            split=args.subset,
            with_flow=False,
            input_transform=input_transform,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, pin_memory=True, shuffle=False
        )

        for data in tqdm(data_loader):
            img1, img2 = data["img1"].cuda(), data["img2"].cuda()
            full_seg1, full_seg2 = data["full_seg1"].cuda(), data["full_seg2"].cuda()

            # compute output
            output = model(img1, img2, full_seg1, full_seg2, with_bk=True)
            flow_pred = output["flows_12"][0]
            flow_pred_bw = output["flows_21"][0]

            for i in range(flow_pred.shape[0]):

                h, w = data["raw_size"][0][i], data["raw_size"][1][i]
                h, w = h.item(), w.item()
                flow_pred_up = resize_flow(flow_pred[i : (i + 1)], (h, w))

                scene, frame_id = data["img1_path"][i].split("/")[-2:]
                filename = frame_id[:5] + frame_id[6:10] + ".flo"
                # output_file = os.path.join(ds_dir, scene, filename)
                output_file_local = os.path.join(ds_dir_local, scene, filename)

                # wrtie to local and then move to manifold
                writeFlowSintel(output_file_local, tensor2array(flow_pred_up)[0])

                ## also compute backward flow
                flow_pred_bw_up = resize_flow(flow_pred_bw[i : (i + 1)], (h, w))
                output_file_local = os.path.join(ds_dir_bw_local, scene, filename)
                writeFlowSintel(output_file_local, tensor2array(flow_pred_bw_up)[0])

                # if not pathmgr.exists(os.path.dirname(output_file)):
                #     pathmgr.mkdirs(os.path.dirname(output_file))
                # pathmgr.copy_from_local(output_file_local, output_file)

    print("Completed!")
    return


@torch.no_grad()
def create_kitti_submission(model, args):
    """Create submission for the KITTI leaderboard"""

    input_transform = transforms.Compose(
        [
            input_transforms.Zoom(args.img_height, args.img_width),
            input_transforms.ArrayToTensor(),
        ]
    )

    dataset_2012 = KITTIFlowEval(
        os.path.join(args.root_kitti12, args.subset + "ing"),
        os.path.join(args.full_seg_root_kitti12, args.subset + "ing"),
        None,
        name="kitti2012",
        input_transform=input_transform,
        test_mode=True,
    )
    dataset_2015 = KITTIFlowEval(
        os.path.join(args.root_kitti15, args.subset + "ing"),
        os.path.join(args.full_seg_root_kitti15, args.subset + "ing"),
        None,
        name="kitti2015",
        input_transform=input_transform,
        test_mode=True,
    )

    # start inference
    model.eval()
    for ds in [dataset_2015, dataset_2012]:
        # ds_dir = os.path.join(args.output_dir, ds.name)
        ds_dir_local = os.path.join(args.output_local_dir, ds.name)
        ds_dir_bw_local = os.path.join(args.output_local_dir + "_bw", ds.name)
        # pathmgr.mkdirs(os.path.join(ds_dir, "flow"))
        os.makedirs(os.path.join(ds_dir_local, "flow"), exist_ok=True)
        os.makedirs(os.path.join(ds_dir_bw_local, "flow"), exist_ok=True)

        data_loader = torch.utils.data.DataLoader(
            ds, batch_size=4, pin_memory=True, shuffle=False
        )
        for data in tqdm(data_loader):

            img1, img2 = data["img1"].cuda(), data["img2"].cuda()
            full_seg1, full_seg2 = data["full_seg1"].cuda(), data["full_seg2"].cuda()

            # compute output
            output = model(img1, img2, full_seg1, full_seg2, with_bk=True)
            flow_pred = output["flows_12"][0]
            flow_pred_bw = output["flows_21"][0]

            for i in range(flow_pred.shape[0]):
                h, w = data["raw_size"][0][i], data["raw_size"][1][i]
                h, w = h.item(), w.item()
                flow_pred_up = resize_flow(flow_pred[i : (i + 1)], (h, w))

                filename = os.path.basename(data["img1_path"][i])
                # output_file = os.path.join(ds_dir, "flow", filename)
                output_file_local = os.path.join(ds_dir_local, "flow", filename)

                # wrtie to local and then move to manifold
                writeFlowKITTI(output_file_local, tensor2array(flow_pred_up)[0])
                # pathmgr.copy_from_local(output_file_local, output_file)

                ## also compute backward flow
                flow_pred_bw_up = resize_flow(flow_pred_bw[i : (i + 1)], (h, w))
                output_file_local = os.path.join(ds_dir_bw_local, "flow", filename)
                writeFlowKITTI(output_file_local, tensor2array(flow_pred_bw_up)[0])

    print("Completed!")
    return


@torch.no_grad()
def main():
    args = parser.parse_args()

    args.full_model_folder = os.path.join(
        "memcache_manifold://", MANIFOLD_BUCKET, MANIFOLD_PATH, args.model_folder
    )

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.full_model_folder, args.subset + "_flow_" + args.dataset
        )
    args.output_local_dir = os.path.join(
        YOUR_DIR,
        args.model_folder,
        args.subset + "_flow_" + args.dataset,
    )

    # pathmgr.mkdirs(args.output_dir)
    os.makedirs(args.output_local_dir, exist_ok=True)

    ## set up the model
    config_file = os.path.join(args.full_model_folder, "config.json")
    model_file = os.path.join(args.full_model_folder, args.trained_model)
    cfg = init_config(config_file)

    model = get_model(cfg.model).cuda()

    model = restore_model(model, model_file)
    model.eval()

    if args.dataset == "sintel":
        args.img_height, args.img_width = 448, 1024
 
        # Use local data to save time
        args.root_sintel = YOUR_DIR
        args.full_seg_root_sintel = YOUR_DIR
        
        create_sintel_submission(model, args)
    elif args.dataset == "kitti":
        args.img_height, args.img_width = 256, 832

        # Use local data to save time
        args.root_kitti12 = YOUR_DIR
        args.root_kitti15 = YOUR_DIR
        args.full_seg_root_kitti12 = YOUR_DIR
        args.full_seg_root_kitti15 = YOUR_DIR

        create_kitti_submission(model, args)


if __name__ == "__main__":
    main()
