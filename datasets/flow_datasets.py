"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import os

from abc import ABCMeta, abstractmethod

# from glob import glob

import imageio
import numpy as np
import torch

# from transforms.input_transforms import full_segs_to_adj_maps
from utils.flow_utils import load_flow

from utils.manifold_utils import pathmgr


def local_path(path):
    if "manifold" in path:
        return pathmgr.get_local_path(path)
    else:
        return path


class ImgSeqDataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    def __init__(
        self,
        root,
        full_seg_root,
        key_obj_root=None,
        name="",
        input_transform=None,
        co_transform=None,
        ap_transform=None,
    ):
        self.root = root
        self.full_seg_root = full_seg_root
        self.key_obj_root = key_obj_root
        self.name = name
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.samples = self.collect_samples()

    @abstractmethod
    def collect_samples(self):
        pass

    def _load_sample(self, s):

        imgs = []
        full_segs = []
        key_objs = []
        for p in s["imgs"]:

            image = (
                imageio.imread(local_path(os.path.join(self.root, p))).astype(
                    np.float32
                )
                / 255.0
            )
            imgs.append(image)

            full_seg = imageio.imread(local_path(os.path.join(self.full_seg_root, p)))[
                :, :, None
            ]
            full_segs.append(full_seg)

            if self.key_obj_root is not None:
                key_obj = (
                    np.load(
                        local_path(os.path.join(self.key_obj_root, p[:-4] + ".npy"))
                    )
                    / 255.0
                )
                key_objs.append(key_obj)

        return imgs, full_segs, key_objs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imgs, full_segs, key_objs = self._load_sample(self.samples[idx])

        data = {
            "raw_size": imgs[0].shape[:2],
            "img1_path": os.path.join(self.root, self.samples[idx]["imgs"][0]),
        }

        if self.co_transform is not None:
            # In unsupervised learning, there is no need to change target with image
            imgs, full_segs, key_objs, _ = self.co_transform(
                imgs, full_segs, key_objs, {}
            )

        if self.input_transform is not None:
            imgs, full_segs, key_objs = self.input_transform(
                (imgs, full_segs, key_objs)
            )

        # adj_maps = full_segs_to_adj_maps(torch.stack(full_segs), win_size=9)

        data.update(
            {
                "img1": imgs[0],
                "img2": imgs[1],
                "full_seg1": full_segs[0],
                "full_seg2": full_segs[1],
            }
        )

        # process key_objs to keep exactly three objects (to make sure the number of objects is fixed so that we can form batches)
        if self.key_obj_root is not None:
            place_holder = torch.full(
                (1, *key_objs[0].shape[1:]), np.nan, dtype=torch.float32
            )

            if key_objs[0].shape[0] == 0:
                key_obj = place_holder
            else:
                valid_key_obj = (
                    key_objs[0].mean(axis=(1, 2)) >= 0.005
                )  ## some objects may be too small after cropping

                if valid_key_obj.sum() == 0:
                    key_obj = place_holder
                else:
                    idx = np.random.choice(np.where(valid_key_obj)[0])
                    key_obj = key_objs[0][idx : idx + 1]

            data["key_obj_mask"] = key_obj

        if self.ap_transform is not None:
            data["img1_ph"], data["img2_ph"] = self.ap_transform(
                [imgs[0].clone(), imgs[1].clone()]
            )

        return data


class KITTIRawFile(ImgSeqDataset):
    def __init__(
        self,
        root,
        full_seg_root,
        key_obj_root,
        name="kitti-raw",
        ap_transform=None,
        input_transform=None,
        co_transform=None,
    ):
        super(KITTIRawFile, self).__init__(
            root,
            full_seg_root,
            key_obj_root,
            name,
            input_transform=input_transform,
            co_transform=co_transform,
            ap_transform=ap_transform,
        )

    def collect_samples(self):
        sp_file = os.path.join(self.root, "kitti_train_2f_sv.txt")

        samples = []
        with open(local_path(sp_file), "r") as f:
            for line in f.readlines():
                sp = line.split()
                samples.append({"imgs": sp[0:2]})
                samples.append({"imgs": sp[2:4]})

        return samples


class KITTIFlowMV(ImgSeqDataset):
    """
    This dataset is used for unsupervised training only
    """

    def __init__(
        self,
        root,
        full_seg_root,
        key_obj_root,
        name="",
        input_transform=None,
        co_transform=None,
        ap_transform=None,
    ):
        super(KITTIFlowMV, self).__init__(
            root,
            full_seg_root,
            key_obj_root,
            name,
            input_transform=input_transform,
            co_transform=co_transform,
            ap_transform=ap_transform,
        )

    def collect_samples(self):

        sp_file = os.path.join(self.root, "sample_list_mv.txt")

        samples = []
        with open(local_path(sp_file), "r") as f:
            for line in f.readlines():
                samples.append({"imgs": line.split()})

        return samples


class KITTIFlowEval(ImgSeqDataset):
    """
    This dataset is used for validation/test ONLY, so all files about target are stored as
    file filepath and there is no transform about target.
    """

    def __init__(
        self,
        root,
        full_seg_root,
        key_obj_root,
        name="",
        input_transform=None,
        test_mode=False,
    ):
        self.test_mode = test_mode
        super(KITTIFlowEval, self).__init__(
            root, full_seg_root, key_obj_root, name, input_transform=input_transform
        )

    def __getitem__(self, idx):
        data = super(KITTIFlowEval, self).__getitem__(idx)
        if not self.test_mode:
            # for validation; we do not load here because different samples have different sizes
            data["flow_occ"] = os.path.join(self.root, self.samples[idx]["flow_occ"])
            data["flow_noc"] = os.path.join(self.root, self.samples[idx]["flow_noc"])

        return data

    def collect_samples(self):
        """Will search in training folder for folders 'flow_noc' or 'flow_occ'
        and 'colored_0' (KITTI 2012) or 'image_2' (KITTI 2015)"""

        sp_file = os.path.join(self.root, "sample_list.txt")

        samples = []
        with open(local_path(sp_file), "r") as f:
            for line in f.readlines():
                samples.append({"imgs": line.split()})

        if self.test_mode:
            return samples
        else:
            for i, sample in enumerate(samples):
                filename = os.path.basename(sample["imgs"][0])

                samples[i].update(
                    {
                        "flow_occ": os.path.join("flow_occ", filename),
                        "flow_noc": os.path.join("flow_noc", filename),
                    }
                )

            return samples


class SintelRaw(ImgSeqDataset):
    def __init__(
        self,
        root,
        full_seg_root,
        key_obj_root,
        name="",
        input_transform=None,
        ap_transform=None,
        co_transform=None,
    ):
        super(SintelRaw, self).__init__(
            root,
            full_seg_root,
            key_obj_root,
            name,
            input_transform=input_transform,
            ap_transform=ap_transform,
            co_transform=co_transform,
        )

    def collect_samples(self):

        sp_file = os.path.join(self.root, "sample_list.txt")

        samples = []
        with open(local_path(sp_file), "r") as f:
            for line in f.readlines():
                samples.append({"imgs": line.split()})

        return samples


class Sintel(ImgSeqDataset):
    def __init__(
        self,
        root,
        full_seg_root,
        key_obj_root,
        name="",
        dataset_type="clean",
        split="train",
        subsplit="trainval",
        with_flow=False,
        input_transform=None,
        co_transform=None,
        ap_transform=None,
    ):
        self.dataset_type = dataset_type
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit
        self.training_scenes = [
            "alley_1",
            "ambush_4",
            "ambush_6",
            "ambush_7",
            "bamboo_2",
            "bandage_2",
            "cave_2",
            "market_2",
            "market_5",
            "shaman_2",
            "sleeping_2",
            "temple_3",
        ]  # Unofficial train-val split

        super(Sintel, self).__init__(
            root,
            full_seg_root,
            key_obj_root,
            name,
            input_transform=input_transform,
            co_transform=co_transform,
            ap_transform=ap_transform,
        )

    def __getitem__(self, idx):
        data = super(Sintel, self).__getitem__(idx)
        if self.with_flow:
            data["flow_gt"] = load_flow(
                pathmgr.get_local_path(self.samples[idx]["flow"])
            ).astype(np.float32)
            data["occ_mask"] = (
                imageio.imread(
                    pathmgr.get_local_path(self.samples[idx]["occ_mask"])
                ).astype(np.float32)[:, :, None]
                / 255.0
            )

        return data

    def collect_samples(self):

        samples = []
        filename = self.split + "_" + self.dataset_type + "_images.txt"
        sp_file = os.path.join(self.root, filename)

        with open(local_path(sp_file), "r") as f:
            for line in f.readlines():
                img1, img2 = line[:-1].split(",")
                path_split = img1.split("/")
                scene = path_split[-2]
                sample = {
                    "imgs": [
                        "/".join(img1.split("/")[-4:]),
                        "/".join(img2.split("/")[-4:]),
                    ]
                }
                if self.with_flow:
                    sample["flow"] = os.path.join(
                        "/".join(path_split[:-3]),
                        "flow",
                        scene,
                        path_split[-1][:-4] + ".flo",
                    )
                    sample["occ_mask"] = os.path.join(
                        "/".join(path_split[:-3]),
                        "occlusions",
                        scene,
                        path_split[-1],
                    )

                if self.subsplit == "trainval":
                    samples.append(sample)
                elif self.subsplit == "train" and scene in self.training_scenes:
                    samples.append(sample)
                elif self.subsplit == "val" and scene not in self.training_scenes:
                    samples.append(sample)

        return samples
