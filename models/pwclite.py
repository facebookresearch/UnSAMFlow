import torch
import torch.nn as nn
import torch.nn.functional as F
from models.correlation_internal import Correlation_ours

# from .correlation_package.correlation import Correlation
# from .correlation_native import Correlation

from transforms.input_transforms import full_segs_to_adj_maps

from utils.warp_utils import flow_warp


def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((kernel_size - 1) * dilation) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((kernel_size - 1) * dilation) // 2,
                bias=True,
            )
        )


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs, input_adj_map=False):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        if input_adj_map:
            self.adj_map_net = nn.Sequential(
                conv(81, 32, kernel_size=1),
                conv(32, 32, kernel_size=3, stride=2),
                conv(32, 32, kernel_size=3),
                conv(32, 32, kernel_size=3, stride=2),
                conv(32, 32, kernel_size=3),
            )
        else:
            self.adj_map_net = None

        for level, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            if input_adj_map and level == 2:
                ch_in += 32
            layer = nn.Sequential(conv(ch_in, ch_out, stride=2), conv(ch_out, ch_out))
            self.convs.append(layer)

    def forward(self, x, adj_map=None):
        feature_pyramid = [x]
        if self.adj_map_net is not None:
            adj_map_feat = self.adj_map_net(adj_map)

        for i, conv in enumerate(self.convs):
            if self.adj_map_net is not None and i == 2:
                x = torch.concat((x, adj_map_feat), dim=1)
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.feat_dim = ch_in + 448
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorReduce(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, ch_in):
        super(FlowEstimatorReduce, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(128 + 96, 64)
        self.conv5 = conv(96 + 64, 32)
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        flow = self.predict_flow(torch.cat([x4, x5], dim=1))
        return x5, flow


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
        )
        self.flow_head = nn.Sequential(
            conv(96, 64, 3, 1, 16), conv(64, 32, 3, 1, 1), conv(32, 2, isReLU=False)
        )

    def forward(self, x):
        feat = self.convs(x)
        flow = self.flow_head(feat)
        return flow, feat


class UpFlowNetwork(nn.Module):
    def __init__(self, ch_in=96, scale_factor=4):
        super(UpFlowNetwork, self).__init__()
        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1), conv(128, scale_factor**2 * 9, 3, 1, 1)
        )

    # adapted from https://github.com/princeton-vl/RAFT/blob/aac9dd54726caf2cf81d8661b07663e220c5586d/core/raft.py#L72
    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4 * H, 4 * W)

    def forward(self, flow, feat):
        # scale mask to balence gradients
        up_mask = 0.25 * self.convs(feat)
        return self.upsample_flow(flow, up_mask)


class PWCLite(nn.Module):
    def __init__(self, cfg):
        super(PWCLite, self).__init__()
        if "input_adj_map" not in cfg:
            cfg.input_adj_map = False

        if "input_boundary" not in cfg:
            cfg.input_boundary = False

        if "add_mask_corr" not in cfg:
            cfg.add_mask_corr = False

        self.cfg = cfg
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 192]
        if cfg.input_boundary:
            self.num_chs[0] += 2

        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        # encoder
        self.feature_pyramid_extractor = FeatureExtractor(
            self.num_chs, input_adj_map=cfg.input_adj_map
        )

        # decoder
        ## Our correlation implementation
        self.corr = Correlation_ours(
            kernel_size=1,
            patch_size=(2 * self.search_range + 1),
            stride=1,
            padding=0,
            dilation_patch=1,
            normalize=True,
        )

        ## Correlation modeuld in the original code
        # self.corr = Correlation(
        #     pad_size=self.search_range,
        #     kernel_size=1,
        #     max_displacement=self.search_range,
        #     stride1=1,
        #     stride2=1,
        #     corr_multiply=1,
        # )

        self.dim_corr = (self.search_range * 2 + 1) ** 2

        if cfg.add_mask_corr:
            self.num_ch_in = 32 + 2 * self.dim_corr + 2
        else:
            self.num_ch_in = 32 + self.dim_corr + 2

        if cfg.reduce_dense:
            self.flow_estimators = FlowEstimatorReduce(self.num_ch_in)
        else:
            self.flow_estimators = FlowEstimatorDense(self.num_ch_in)

        self.context_networks = ContextNetwork(self.flow_estimators.feat_dim + 2)

        if cfg.learned_upsampler:
            self.output_flow_upsampler = UpFlowNetwork(ch_in=96, scale_factor=4)
        else:
            self.output_flow_upsampler = None

        self.conv_1x1 = nn.ModuleList(
            [
                conv(self.num_chs[-1], 32, kernel_size=1, stride=1, dilation=1),
                conv(self.num_chs[-2], 32, kernel_size=1, stride=1, dilation=1),
                conv(self.num_chs[-3], 32, kernel_size=1, stride=1, dilation=1),
                conv(self.num_chs[-4], 32, kernel_size=1, stride=1, dilation=1),
                conv(self.num_chs[-5], 32, kernel_size=1, stride=1, dilation=1),
            ]
        )

        if cfg.add_mask_corr:
            self.conv_1x1_mask = nn.ModuleList(
                [
                    conv(self.num_chs[-1], 32, kernel_size=1, stride=1, dilation=1),
                    conv(self.num_chs[-2], 32, kernel_size=1, stride=1, dilation=1),
                    conv(self.num_chs[-3], 32, kernel_size=1, stride=1, dilation=1),
                    conv(self.num_chs[-4], 32, kernel_size=1, stride=1, dilation=1),
                    conv(self.num_chs[-5], 32, kernel_size=1, stride=1, dilation=1),
                ]
            )

            if self.cfg.aggregation_type == "residual":
                self.mask_aggregation = conv(
                    32, 32, kernel_size=1, stride=1, dilation=1
                )
            elif self.cfg.aggregation_type == "concat":
                self.mask_aggregation = conv(
                    64, 32, kernel_size=1, stride=1, dilation=1
                )

    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()]
        )

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def decoder(self, x1_pyramid, x2_pyramid, full_seg1=None, full_seg2=None):
        # outputs
        flows = []

        # init
        (
            b_size,
            _,
            h_x1,
            w_x1,
        ) = x1_pyramid[0].size()
        init_dtype = x1_pyramid[0].dtype
        init_device = x1_pyramid[0].device
        flow = torch.zeros(
            b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device
        ).float()

        for level, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if level > 0:
                flow = F.interpolate(
                    flow * 2, scale_factor=2, mode="bilinear", align_corners=True
                )
                x2_warp = flow_warp(x2, flow)
            else:
                x2_warp = x2

            # correlation
            out_corr = self.corr(x1, x2_warp)
            out_corr_relu = self.leakyRELU(out_corr)

            # import IPython

            # IPython.embed()
            # exit()

            x1_1by1 = self.conv_1x1[level](x1)

            if self.cfg.add_mask_corr:
                x1_1by1_mask = self.conv_1x1_mask[level](x1)
                full_seg1_down = F.interpolate(
                    full_seg1, x1.shape[-2:], mode="nearest"
                ).long()
                full_seg1_down_oh = F.one_hot(full_seg1_down)
                mask_pooled_value1 = torch.amax(
                    full_seg1_down_oh * x1_1by1_mask[..., None], dim=(2, 3)
                )
                mask_feat1 = (
                    full_seg1_down_oh * mask_pooled_value1[:, :, None, None, :]
                ).sum(dim=-1)

                x2_1by1_mask = self.conv_1x1_mask[level](x2)
                full_seg2_down = F.interpolate(
                    full_seg2, x2.shape[-2:], mode="nearest"
                ).long()
                full_seg2_down_oh = F.one_hot(full_seg2_down)
                mask_pooled_value2 = torch.amax(
                    full_seg2_down_oh * x2_1by1_mask[..., None], dim=(2, 3)
                )
                mask_feat2 = (
                    full_seg2_down_oh * mask_pooled_value2[:, :, None, None, :]
                ).sum(dim=-1)

                if self.cfg.aggregation_type == "residual":
                    x_mask_feat1 = x1_1by1_mask + self.mask_aggregation(mask_feat1)
                    x_mask_feat2 = x2_1by1_mask + self.mask_aggregation(mask_feat2)
                elif self.cfg.aggregation_type == "concat":
                    x_mask_feat1 = self.mask_aggregation(
                        torch.concat((x1_1by1_mask, mask_feat1), axis=1)
                    )
                    x_mask_feat2 = self.mask_aggregation(
                        torch.concat((x2_1by1_mask, mask_feat2), axis=1)
                    )
                else:
                    raise NotImplementedError

                x_mask_feat2_warp = flow_warp(x_mask_feat2, flow)
                out_mask_corr = self.corr(x_mask_feat1, x_mask_feat2_warp)
                out_mask_corr_relu = self.leakyRELU(out_mask_corr)

                x_intm, flow_res = self.flow_estimators(
                    torch.cat([out_corr_relu, out_mask_corr_relu, x1_1by1, flow], dim=1)
                )

            else:
                x_intm, flow_res = self.flow_estimators(
                    torch.cat([out_corr_relu, x1_1by1, flow], dim=1)
                )

            flow = flow + flow_res

            flow_fine, up_feat = self.context_networks(torch.cat([x_intm, flow], dim=1))
            flow = flow + flow_fine

            if self.output_flow_upsampler is not None:
                flow_up = self.output_flow_upsampler(flow, up_feat)
            else:
                flow_up = F.interpolate(
                    flow * 4, scale_factor=4, mode="bilinear", align_corners=True
                )
            flows.append(flow_up)

            # upsampling or post-processing
            if level == self.output_level:
                break

        return flows[::-1]

    def forward(self, img1, img2, full_seg1=None, full_seg2=None, with_bk=False):

        batch_size, _, h, w = img1.shape

        if self.cfg.input_adj_map:
            adj_maps = full_segs_to_adj_maps(
                torch.concat((full_seg1, full_seg2), axis=0)
            )
            adj_map1 = adj_maps[:batch_size]
            adj_map2 = adj_maps[batch_size:]
        else:
            adj_map1, adj_map2 = None, None

        if self.cfg.input_boundary:

            def compute_seg_edge(full_seg):
                batch_size, _, h, w = full_seg.shape
                seg_edge_x = (full_seg[..., :, 1:] != full_seg[..., :, :-1]).float()
                seg_edge_x = torch.concat(
                    (
                        seg_edge_x,
                        torch.zeros((batch_size, 1, h, 1)).to(seg_edge_x.device),
                    ),
                    axis=-1,
                )
                seg_edge_y = (full_seg[..., 1:, :] != full_seg[..., :-1, :]).float()
                seg_edge_y = torch.concat(
                    (
                        seg_edge_y,
                        torch.zeros((batch_size, 1, 1, w)).to(seg_edge_x.device),
                    ),
                    axis=-2,
                )
                return seg_edge_x, seg_edge_y

            img1 = torch.concat((img1, *compute_seg_edge(full_seg1)), axis=1)
            img2 = torch.concat((img2, *compute_seg_edge(full_seg2)), axis=1)

        feat1 = self.feature_pyramid_extractor(img1, adj_map1)
        feat2 = self.feature_pyramid_extractor(img2, adj_map2)

        # decode outputs
        res_dict = {}
        res_dict["flows_12"] = self.decoder(feat1, feat2, full_seg1, full_seg2)
        if with_bk:
            res_dict["flows_21"] = self.decoder(feat2, feat1, full_seg2, full_seg1)

        return res_dict
