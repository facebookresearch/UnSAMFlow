"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

# Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = (
            image[:, 0, :, :] * 0.2989
            + image[:, 1, :, :] * 0.5870
            + image[:, 2, :, :] * 0.1140
        )
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask


def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist


def gradient(data):
    D_dy = data[..., 1:, :] - data[..., :-1, :]
    D_dx = data[..., :, 1:] - data[..., :, :-1]
    return D_dx, D_dy


def get_image_edge_weights(image, alpha=10):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
    return weights_x, weights_y


def get_full_seg_edge_weights(full_seg):
    weights_y = (full_seg[..., 1:, :] - full_seg[..., :-1, :] == 0).float()
    weights_x = (full_seg[..., :, 1:] - full_seg[..., :, :-1] == 0).float()
    return weights_x, weights_y


def smooth_grad_1st(flo, image, edge="image", **kwargs):
    if edge == "image":
        weights_x, weights_y = get_image_edge_weights(image, kwargs["alpha"])
    elif edge == "full_seg":
        weights_x, weights_y = get_full_seg_edge_weights(kwargs["full_seg"])

    dx, dy = gradient(flo)
    loss_x = weights_x * dx.abs()
    loss_y = weights_y * dy.abs()

    return loss_x.mean() / 2.0 + loss_y.mean() / 2.0


def smooth_grad_2nd(flo, image, edge="image", **kwargs):
    if edge == "image":
        weights_x, weights_y = get_image_edge_weights(image, kwargs["alpha"])
    elif edge == "full_seg":
        weights_x, weights_y = get_full_seg_edge_weights(kwargs["full_seg"])

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()
    # loss_x = weights_x[:, :, :, 1:] * (torch.exp(dx2.abs() * 100) - 1) / 100.
    # loss_y = weights_y[:, :, 1:, :] * (torch.exp(dy2.abs() * 100) - 1) / 100.

    return loss_x.mean() / 2.0 + loss_y.mean() / 2.0


def smooth_homography(flo, full_seg, occ_mask, ransac_threshold=3):

    DEVICE = flo.device
    B, _, h, w = flo.shape

    loss = torch.tensor(0, dtype=torch.float32, device=DEVICE)
    for i in range(B):

        ## find regions to refine
        n = int(full_seg[i].max().item() + 1)
        occ_mask_ids = full_seg[i, occ_mask[i].to(bool)].to(int)
        occ_mask_id_count = torch.eye(n, dtype=bool, device=DEVICE)[occ_mask_ids].sum(
            axis=0
        )

        id_order = occ_mask_id_count.argsort(descending=True)
        refine_id = id_order[id_order > 0][
            :6
        ]  # we disregard the `0` mask id because it is just the non-masked region, not one object
        refine_id = refine_id.tolist()

        ## start refining
        coords1 = (
            torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w))[::-1], axis=2)
            .float()
            .to(DEVICE)
        )
        coords2 = coords1 + flo[i].permute(1, 2, 0)

        # effective_mask_ids = []
        for id in refine_id:
            reliable_mask = (
                (1 - occ_mask[i, full_seg[i] == id]).bool().detach().cpu().numpy()
            )
            if reliable_mask.sum() < 4 or reliable_mask.mean() < 0.2:
                # print("Mask #{0:} dropped due to low non-occcluded ratio ({1:4.2f}).".format(id, reliable_mask.mean()))
                continue

            pts1 = coords1[full_seg[i, 0] == id]
            pts2 = coords2[full_seg[i, 0] == id]

            H, mask = cv2.findHomography(
                pts1[reliable_mask].detach().cpu().numpy(),
                pts2[reliable_mask].detach().cpu().numpy(),
                cv2.RANSAC,
                ransac_threshold,
            )

            if (
                mask.mean() < 0.5
            ):  # do not refine if the estimated homography's inlier rate < 0.5
                # print("Mask #{0:} dropped due to low inlier rate ({1:4.2f}).".format(id, mask.mean()))
                continue

            H = torch.FloatTensor(H).to(DEVICE)
            pts1_homo = torch.concat(
                (pts1, torch.ones((pts1.shape[0], 1)).to(DEVICE)), dim=1
            )
            new_pts2_homo = torch.matmul(H, pts1_homo.T).T
            new_pts2 = new_pts2_homo[:, :2] / new_pts2_homo[:, 2:3]
            diff = (new_pts2 - pts2)[:, :2]
            # flow_refined[i, :, full_seg[i, 0] == id] = (new_pts2 - pts1)[:, :2].T

            loss += diff.abs().sum() / (h * w)
            # effective_mask_ids.append(id)

            ## DEBUG:
            # import IPython; IPython.embed(); exit()
            # import matplotlib.pyplot as plt
            # plt.imsave("_DEBUG_flow.png", flow_to_image(flo[i].detach().cpu().numpy().transpose(1, 2, 0)))
            # plt.imsave("_DEBUG_flow_refined.png", flow_to_image(flow_refined[i].detach().cpu().numpy().transpose(1, 2, 0)))
            # from skimage.segmentation import mark_boundaries
            # plt.imsave("_DEBUG_full_seg.png", mark_boundaries(occ_mask[i].detach().cpu().numpy().transpose(1, 2, 0).squeeze(), full_seg[i].detach().cpu().numpy().transpose(1, 2, 0).squeeze().astype(int)))

    loss /= B
    return loss
