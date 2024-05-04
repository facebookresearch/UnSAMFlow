"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Adapted from https://github.com/facebookresearch/segment-anything/blob/main/scripts/amg.py
"""

import argparse
import json
import os
from typing import Any, Dict, List

import cv2  # type: ignore

# from utils.manifold_utils import pathmgr
import numpy as np

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

# parser.add_argument(
#     "--input",
#     type=str,
#     required=True,
#     help="Path to either a single input image or folder of images.",
# )

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="The dataset for inference.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument(
    "--device", type=str, default="cuda", help="The device to run generation on."
)

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:

    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    # if not os.path.isdir(args.input):
    #     targets = [args.input]
    # else:
    #     targets = [
    #         f
    #         for f in os.listdir(args.input)
    #         if not os.path.isdir(os.path.join(args.input, f))
    #     ]
    #     targets = [os.path.join(args.input, f) for f in targets]

    if args.dataset == "KITTI-2015" or args.dataset == "KITTI-2012":
        dataset_root = (
            YOUR_DIR + args.dataset
        )

        targets = []
        for split in ["training", "testing"]:
            with open(os.path.join(dataset_root, split, "image_list.txt"), "r") as f:
                line = f.readlines()[0]
                line = line.split(" ")
                targets += [os.path.join(split, t) for t in line]

    elif args.dataset == "KITTI-raw":
        dataset_root = YOUR_DIR

        targets = []
        with open(os.path.join(dataset_root, "kitti_train_2f_sv.txt"), "r") as f:
            lines = f.readlines()

        for line in lines:
            targets += line.split()
        targets = np.unique(targets).tolist()

    elif args.dataset == "Sintel":
        dataset_root = YOUR_DIR

        targets = []
        for split in ["training", "test"]:
            with open(os.path.join(dataset_root, split, "image_list.txt"), "r") as f:
                line = f.readlines()[0]
                line = line.split(" ")
                targets += [os.path.join(split, t) for t in line]

    elif args.dataset == "Sintel-raw":
        dataset_root = YOUR_DIR

        targets = []
        with open(os.path.join(dataset_root, "sample_list.txt"), "r") as f:
            lines = f.readlines()

        for line in lines:
            targets += line.split()
        targets = np.unique(targets).tolist()

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    os.makedirs(os.path.join(args.output, args.dataset), exist_ok=True)

    for t in tqdm(targets):
        print(f"Processing '{t}'...")
        image = cv2.imread(os.path.join(dataset_root, t))
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)

        base = os.path.splitext(t)[0]
        save_base = os.path.join(args.output, args.dataset, base)
        if not os.path.exists(os.path.dirname(save_base)):
            os.makedirs(os.path.dirname(save_base), exist_ok=True)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=False)
            write_masks_to_folder(masks, save_base)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)
    print("Done!")


def main_mask_to_full_seg():
    home_dir = YOUR_DIR

    import imageio
    from pycocotools import mask as mask_utils

    # ds = "KITTI-raw"
    # ds = "Sintel-raw"
    ds = "KITTI-2012/training"

    with open("{}/data/{}/image_list_mv.txt".format(home_dir, ds), "r") as f:
        lines = f.readlines()
        img_list = [line.strip() for line in lines]

    for img_name in tqdm(img_list):
        with open(
            "{}/results/sam_results/raw/{}/{}.json".format(home_dir, ds, img_name[:-4]),
            "r",
        ) as f:
            masks = json.load(f)

        masks_map = np.array(
            mask_utils.decode([mask["segmentation"] for mask in masks]),
            dtype=np.float32,
        )

        H, W = masks_map.shape[:2]
        masks_area = np.array([mask["area"] for mask in masks])

        # drop mask if it equals the full frame
        masks_map = masks_map[:, :, masks_area < H * W]
        masks_area = masks_area[masks_area < H * W]

        # sort the class ids by area, largest to smallest
        area_order = np.argsort(masks_area)[::-1]
        masks_area = masks_area[area_order]
        masks_map = masks_map[:, :, area_order]

        # add a "background mask" for pixels that are not included in any masks
        masks_map_aug = np.concatenate((np.ones((H, W, 1)), masks_map), axis=-1)
        masks_area_aug = np.array([H * W] + masks_area.tolist())
        masks_area_aug = np.array(masks_area_aug, dtype=np.float32)

        unified_mask = np.argmin(
            masks_map_aug * masks_area_aug[None, None, :]
            + (1 - masks_map_aug) * (H * W + 1),
            axis=-1,
        )

        unique_classes = np.unique(unified_mask)
        mapping = np.zeros((unique_classes.max() + 1))
        for i, cl in enumerate(unique_classes):
            mapping[cl] = i
        new_mask = mapping[unified_mask]

        if new_mask.max() > 255:  # almost not existent
            print("More than 256 masks detect for image {}".format(img_name))
            new_mask[new_mask > 255] = 0
        new_mask = new_mask.astype(np.uint8)

        save_path = "{}/results/sam_results/full_seg/{}/{}".format(
            home_dir, ds, img_name
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.imwrite(save_path, new_mask)


def main_mask_to_key_objects():
    home_dir = YOUR_DIR

    from pycocotools import mask as mask_utils

    # ds = "KITTI-raw"
    # ds = "Sintel-raw"
    # ds = "KITTI-2015/training"
    ds = "KITTI-2012/training"
    # ds = "Sintel/training"

    with open("{}/data/{}/image_list_mv.txt".format(home_dir, ds), "r") as f:
        lines = f.readlines()
        img_list = [line.strip() for line in lines]

    for img_name in tqdm(img_list):
        with open(
            "{}/results/sam_results/raw/{}/{}.json".format(home_dir, ds, img_name[:-4]),
            "r",
        ) as f:
            masks = json.load(f)

        masks_map = np.array(
            mask_utils.decode([mask["segmentation"] for mask in masks]),
            dtype=np.float32,
        )
        H, W = masks_map.shape[:2]
        obj_masks = np.zeros((H, W, 0), dtype=np.uint8)

        for mask_id in range(len(masks)):
            mask = masks_map[:, :, mask_id]
            w, h = masks[mask_id]["bbox"][2:4]
            area = masks[mask_id]["area"]

            if not (50 <= h <= 200 and 50 <= w <= 300):
                continue

            if area / (h * w) < 0.5:
                continue

            num_unique_masks = ((masks_map * mask[:, :, None]).sum((0, 1)) > 0).sum()
            if num_unique_masks >= 6:
                obj_masks = np.concatenate(
                    (obj_masks, (mask[:, :, None] * 255).astype(np.uint8)), axis=-1
                )

        save_path = "{}/results/sam_results/key_objects/{}/{}.npy".format(
            home_dir, ds, img_name[:-4]
        )
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.save(save_path, obj_masks)


def invoke_main() -> None:
    args = parser.parse_args()
    main(args)

    # main_mask_to_full_seg()

    # main_mask_to_key_objects()


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
