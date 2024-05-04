"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchvision
from utils.warp_utils import (
    flow_warp,
    get_occu_mask_backward,
    get_occu_mask_bidirection,
)

from .loss_blocks import (
    smooth_grad_1st,
    smooth_grad_2nd,
    smooth_homography,
    SSIM,
    TernaryLoss,
)


class unFlowLoss(nn.modules.Module):
    def __init__(self, cfg):
        super(unFlowLoss, self).__init__()
        self.cfg = cfg
        if "ransac_threshold" not in cfg:
            self.cfg.ransac_threshold = 3

    def loss_photomatric(self, im1_scaled, im1_recons, vis_mask1):
        loss = []

        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * vis_mask1]

        if self.cfg.w_ssim > 0:
            loss += [
                self.cfg.w_ssim * SSIM(im1_recons * vis_mask1, im1_scaled * vis_mask1)
            ]

        if self.cfg.w_ternary > 0:
            loss += [
                self.cfg.w_ternary
                * TernaryLoss(im1_recons * vis_mask1, im1_scaled * vis_mask1)
            ]

        return sum([item.mean() for item in loss]) / (vis_mask1.mean() + 1e-6)

    def loss_smooth(self, flow, im1_scaled, **kwargs):

        loss = []
        if self.cfg.smooth_type == "2nd":
            func_smooth = smooth_grad_2nd
        elif self.cfg.smooth_type == "1st":
            func_smooth = smooth_grad_1st

        if "smooth_edge" not in self.cfg or self.cfg.smooth_edge == "image":
            loss += [
                func_smooth(
                    flow, im1_scaled, edge="image", alpha=self.cfg.edge_aware_alpha
                )
            ]
        else:
            loss += [
                func_smooth(
                    flow, im1_scaled, edge="full_seg", full_seg=kwargs["full_seg"]
                )
            ]
        return sum([item.mean() for item in loss])

    def loss_smooth_homography(self, flow, full_seg, occ_mask):
        loss = smooth_homography(
            flow,
            full_seg=full_seg,
            occ_mask=occ_mask,
            ransac_threshold=self.cfg.ransac_threshold,
        )
        return loss

    def loss_one_pair(
        self, pyramid_flows, im1_origin, im2_origin, occ_aware=True, **kwargs
    ):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """
        DEVICE = pyramid_flows[0].device

        # process data
        B, _, H, W = im1_origin.shape

        # generate visibility mask/occlusion estimation
        top_flow = pyramid_flows[0]
        scale = min(*top_flow.shape[-2:])

        if self.cfg.occ_from_back:
            vis_mask1 = 1 - get_occu_mask_backward(top_flow[:, 2:], th=0.2)
            vis_mask2 = 1 - get_occu_mask_backward(top_flow[:, :2], th=0.2)
        else:
            vis_mask1 = 1 - get_occu_mask_bidirection(top_flow[:, :2], top_flow[:, 2:])
            vis_mask2 = 1 - get_occu_mask_bidirection(top_flow[:, 2:], top_flow[:, :2])

        pyramid_vis_mask1 = [vis_mask1]
        pyramid_vis_mask2 = [vis_mask2]
        for i in range(1, 5):
            _, _, h, w = pyramid_flows[i].size()
            pyramid_vis_mask1.append(F.interpolate(vis_mask1, (h, w), mode="nearest"))
            pyramid_vis_mask2.append(F.interpolate(vis_mask2, (h, w), mode="nearest"))

        # compute losses at each level
        pyramid_warp_losses = []
        pyramid_smooth_losses = []
        zero_loss = torch.tensor(0, dtype=torch.float32, device=DEVICE)

        for i, flow in enumerate(pyramid_flows):

            # resize images to match the size of layer
            b, _, h, w = flow.size()
            im1_scaled, im2_scaled = None, None

            # photometric loss
            if self.cfg.w_ph_scales[i] > 0:
                im1_scaled = F.interpolate(im1_origin, (h, w), mode="area")
                im2_scaled = F.interpolate(im2_origin, (h, w), mode="area")
                im1_recons = flow_warp(im2_scaled, flow[:, :2], pad=self.cfg.warp_pad)
                im2_recons = flow_warp(im1_scaled, flow[:, 2:], pad=self.cfg.warp_pad)

                if occ_aware:
                    vis_mask1, vis_mask2 = pyramid_vis_mask1[i], pyramid_vis_mask2[i]
                else:
                    vis_mask1 = torch.ones(
                        (b, 1, h, w), dtype=torch.float32, device=DEVICE
                    )
                    vis_mask2 = torch.ones(
                        (b, 1, h, w), dtype=torch.float32, device=DEVICE
                    )

                loss_warp = self.loss_photomatric(im1_scaled, im1_recons, vis_mask1)
                if self.cfg.with_bk:
                    loss_warp += self.loss_photomatric(
                        im2_scaled, im2_recons, vis_mask2
                    )
                    loss_warp /= 2.0
                pyramid_warp_losses.append(loss_warp)

            else:
                pyramid_warp_losses.append(zero_loss)

            # smoothness loss
            if i == 0 and self.cfg.w_sm > 0:
                if self.cfg.smooth_type == "homography":
                    loss_smooth = self.loss_smooth_homography(
                        flow[:, :2],
                        full_seg=kwargs["full_seg1"],
                        occ_mask=1 - vis_mask1,
                    )
                    if self.cfg.with_bk:
                        loss_smooth += self.loss_smooth_homography(
                            flow[:, 2:],
                            full_seg=kwargs["full_seg2"],
                            occ_mask=1 - vis_mask2,
                        )
                        loss_smooth /= 2.0
                else:
                    if im1_scaled is None:
                        im1_scaled = F.interpolate(im1_origin, (h, w), mode="area")
                        im2_scaled = F.interpolate(im2_origin, (h, w), mode="area")

                    loss_smooth = self.loss_smooth(
                        flow[:, :2] / scale, im1_scaled, full_seg=kwargs["full_seg1"]
                    )
                    if self.cfg.with_bk:
                        loss_smooth += self.loss_smooth(
                            flow[:, 2:] / scale,
                            im2_scaled,
                            full_seg=kwargs["full_seg2"],
                        )
                        loss_smooth /= 2.0

                pyramid_smooth_losses.append(loss_smooth)

            else:
                pyramid_smooth_losses.append(zero_loss)

            # debug: print to see
            """
            import numpy as np
            from utils.flow_utils import flow_to_image
            import matplotlib.pyplot as plt

            img1_show = (im1_scaled.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            img2_show = (im2_scaled.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

            flow12_numpy = flow[:, :2].detach().cpu().numpy().transpose(0, 2, 3, 1)
            flow12_show = []
            for f in flow12_numpy:
                flow12_show.append(flow_to_image(f))
            flow12_show = np.stack(flow12_show)

            flow21_numpy = flow[:, 2:].detach().cpu().numpy().transpose(0, 2, 3, 1)
            flow21_show = []
            for f in flow21_numpy:
                flow21_show.append(flow_to_image(f))
            flow21_show = np.stack(flow21_show)

            vis1_show = (vis_mask1.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
            vis2_show = (vis_mask2.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
            img1_warp_show = (im1_recons.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            img2_warp_show = (im2_recons.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

            ternary12, ternary21, sem12, sem21 = TEMP[-4:]
            ternary12_show = (ternary12.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
            ternary21_show = (ternary21.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
            sem12_show = ((sem12/2).detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)
            sem21_show = ((sem21/2).detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8).repeat(3, axis=3)

            all_show = np.concatenate((np.concatenate((img1_show, img2_show, sem1_show, sem2_show), axis=1),
                                       np.concatenate((flow12_show, flow21_show, vis1_show, vis2_show), axis=1),
                                       np.concatenate((img1_warp_show, img2_warp_show, sem1_warp_show, sem2_warp_show), axis=1),
                                       np.concatenate((ternary12_show, ternary21_show, sem12_show, sem21_show), axis=1)),
                                      axis=2)
            b, h, w, c = all_show.shape
            all_show = np.concatenate((all_show[:, :, :w//2, :], all_show[:, :, w//2:, :]), axis=1)
            #all_show = all_show.reshape((b*h, w, c))
            all_show = all_show[0]

            import IPython; IPython.embed(); exit()
            plt.imsave('_DEBUG_DEMO_{}.png'.format(i), all_show)
            """

            """
            if i == 0:  # for analysis
                self.l_ph_0 = loss_warp
                self.l_ph_L1_map_0 = (im1_scaled - im1_recons).abs().mean(dim=1)
            """

        # aggregate losses
        pyramid_warp_losses = [
            item * w for item, w in zip(pyramid_warp_losses, self.cfg.w_ph_scales)
        ]

        l_ph = sum(pyramid_warp_losses)
        l_sm = sum(pyramid_smooth_losses)

        loss = l_ph + self.cfg.w_sm * l_sm

        return (
            loss,
            l_ph,
            l_sm,
            pyramid_flows[0][:, :2].norm(dim=1).mean(),
            pyramid_vis_mask1[0],
            pyramid_vis_mask2[0],
        )

    def forward(self, pyramid_flows, img1, img2, occ_aware=True, **kwargs):
        (
            loss,
            l_ph,
            l_sm,
            flow_mean,
            flow_vis_mask12,
            flow_vis_mask21,
        ) = self.loss_one_pair(pyramid_flows, img1, img2, occ_aware=occ_aware, **kwargs)

        return (
            loss[None],
            l_ph[None],
            l_sm[None],
            flow_mean[None],
            flow_vis_mask12,
            flow_vis_mask21,
        )
