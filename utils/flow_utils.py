import os

import cv2
import imageio
import numpy as np
import torch


def load_flow(path):
    if path.endswith(".png"):
        # for KITTI which uses 16bit PNG images
        # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
        # The -1 is here to specify not to change the image depth (16bit), and is compatible
        # with both OpenCV2 and OpenCV3
        flo_file = cv2.imread(path, -1)
        flo_img = flo_file[:, :, 2:0:-1].astype(np.float32)
        invalid = flo_file[:, :, 0] == 0  # mask
        flo_img = flo_img - 32768
        flo_img = flo_img / 64
        flo_img[np.abs(flo_img) < 1e-10] = 1e-10
        flo_img[invalid, :] = 0
        return flo_img, np.expand_dims(flo_file[:, :, 0], 2)
    else:
        with open(path, "rb") as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert 202021.25 == magic, "Magic number incorrect. Invalid .flo file"
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (w, h, 2))
        return data2D


def load_mask(path):
    # 0~255 HxWx1
    mask = imageio.imread(path).astype(np.float32) / 255.0
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    return np.expand_dims(mask, -1)


# def flow_to_image(flow, max_flow=256):
#     import numpy as np
#     from matplotlib.colors import hsv_to_rgb
#     if max_flow is not None:
#         max_flow = max(max_flow, 1.0)
#     else:
#         max_flow = np.max(flow)

#     n = 8
#     u, v = flow[:, :, 0], flow[:, :, 1]
#     mag = np.sqrt(np.square(u) + np.square(v))
#     angle = np.arctan2(v, u)
#     im_h = np.mod(angle / (2 * np.pi) + 1, 1)
#     im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
#     im_v = np.clip(n - im_s, a_min=0, a_max=1)
#     im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
#     return (im * 255).astype(np.uint8)


def resize_flow(flow, new_shape):
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = torch.nn.functional.interpolate(
        flow, (new_h, new_w), mode="bilinear", align_corners=True
    )
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow


# credit: https://github.com/princeton-vl/RAFT/blob/master/core/utils/frame_utils.py
def writeFlowSintel(filename, uv, v=None):
    """Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2
    TAG_CHAR = np.array([202021.25], np.float32)

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        # write the header
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)


# credit: https://github.com/princeton-vl/RAFT/blob/master/core/utils/frame_utils.py
def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def evaluate_flow(gt_flows, pred_flows, moving_masks=None):
    # credit "undepthflow/eval/evaluate_flow.py"
    def calculate_error_rate(epe_map, gt_flow, mask):
        bad_pixels = np.logical_and(
            epe_map * mask > 3,
            epe_map * mask > 0.05 * np.sqrt(np.sum(np.square(gt_flow), axis=2)),
        )
        return bad_pixels.sum() / mask.sum() * 100.0

    (
        error,
        error_noc,
        error_occ,
        error_move,
        error_static,
        error_rate,
        error_rate_noc,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    error_move_rate, error_static_rate = 0.0, 0.0
    B = len(gt_flows)
    for gt_flow, pred_flow, i in zip(gt_flows, pred_flows, range(B)):
        H, W = gt_flow.shape[:2]
        h, w = pred_flow.shape[:2]

        # pred_flow = np.copy(pred_flow)
        # pred_flow[:, :, 0] = pred_flow[:, :, 0] / w * W
        # pred_flow[:, :, 1] = pred_flow[:, :, 1] / h * H

        # flo_pred = cv2.resize(pred_flow, (W, H), interpolation=cv2.INTER_LINEAR)

        pred_flow = torch.from_numpy(pred_flow)[None].permute(0, 3, 1, 2)
        flo_pred = resize_flow(pred_flow, (H, W))
        flo_pred = flo_pred[0].numpy().transpose(1, 2, 0)

        epe_map = np.sqrt(
            np.sum(np.square(flo_pred[:, :, :2] - gt_flow[:, :, :2]), axis=2)
        )
        if gt_flow.shape[-1] == 2:
            error += np.mean(epe_map)

        elif gt_flow.shape[-1] == 4:  # with occ and noc mask
            error += np.sum(epe_map * gt_flow[:, :, 2]) / np.sum(gt_flow[:, :, 2])
            noc_mask = gt_flow[:, :, -1]
            error_noc += np.sum(epe_map * noc_mask) / np.sum(noc_mask)

            error_occ += np.sum(epe_map * (gt_flow[:, :, 2] - noc_mask)) / max(
                np.sum(gt_flow[:, :, 2] - noc_mask), 1.0
            )

            error_rate += calculate_error_rate(
                epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2]
            )
            error_rate_noc += calculate_error_rate(
                epe_map, gt_flow[:, :, 0:2], noc_mask
            )
            if moving_masks is not None:
                move_mask = moving_masks[i]

                error_move_rate += calculate_error_rate(
                    epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2] * move_mask
                )
                error_static_rate += calculate_error_rate(
                    epe_map, gt_flow[:, :, 0:2], gt_flow[:, :, 2] * (1.0 - move_mask)
                )

                error_move += np.sum(epe_map * gt_flow[:, :, 2] * move_mask) / np.sum(
                    gt_flow[:, :, 2] * move_mask
                )
                error_static += np.sum(
                    epe_map * gt_flow[:, :, 2] * (1.0 - move_mask)
                ) / np.sum(gt_flow[:, :, 2] * (1.0 - move_mask))

    if gt_flows[0].shape[-1] == 4:
        res = [
            error / B,
            error_noc / B,
            error_occ / B,
            error_rate / B,
            error_rate_noc / B,
        ]
        if moving_masks is not None:
            res += [error_move / B, error_static / B]
        return res
    else:
        return [error / B]
