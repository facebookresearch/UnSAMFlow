"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import time
from copy import deepcopy

import numpy as np

import torch
from transforms.ar_transforms.oc_transforms import add_fake_object, random_crop
from transforms.ar_transforms.sp_transforms import RandomAffineFlow

# from transforms.input_transforms import full_segs_to_adj_maps
from utils.flow_utils import evaluate_flow
from utils.misc_utils import AverageMeter

from .base_trainer import BaseTrainer


class TrainFramework(BaseTrainer):
    def __init__(
        self,
        train_loader,
        valid_loader,
        model,
        loss_func,
        save_root,
        config,
        resume=False,
        train_sets_epoches=None,
        summary_writer=None,
        rank=0,
        world_size=1,
    ):
        super(TrainFramework, self).__init__(
            train_loader,
            valid_loader,
            model,
            loss_func,
            save_root,
            config,
            resume=resume,
            train_sets_epoches=train_sets_epoches,
            summary_writer=summary_writer,
            rank=rank,
            world_size=world_size,
        )
        self.sp_transform = RandomAffineFlow(
            self.cfg.st_cfg, addnoise=self.cfg.st_cfg.add_noise
        ).to(self.device)

    def _run_one_epoch(self):

        self.model.train()

        if "stage1" in self.cfg and self.i_epoch >= self.cfg.stage1.epoch:
            self.loss_func.cfg.update(self.cfg.stage1.loss)
            self.cfg.update(self.cfg.stage1.train)
            self.cfg.pop("stage1")

        if "stage2" in self.cfg and self.i_epoch >= self.cfg.stage2.epoch:
            if "loss" in self.cfg.stage2:
                self.loss_func.cfg.update(self.cfg.stage2.loss)
            if "train" in self.cfg.stage2:
                self.cfg.update(self.cfg.stage2.train)
            if self.cfg.key_obj_aug:
                self.set_up_obj_cache(cache_size=100)
            self.cfg.pop("stage2")

        timing_meter_names = [
            "1_data_loading",
            "2_main_forward",
            "3_main_loss",
            "4_atst",
            "5_ot",
            "6_bakward_update",
        ]
        timing_meters = AverageMeter(i=len(timing_meter_names))
        key_meter_names = ["loss", "l_ph", "l_sm", "l_atst", "l_ot", "flow_mean"]
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)

        name_dataset = self.train_loaders[self.i_train_set].dataset.name
        last_time = time.time()

        self.train_loaders[self.i_train_set].sampler.set_epoch(self.i_epoch)

        i_step = 0
        while i_step < self.cfg.epoch_size:
            for data in self.train_loaders[self.i_train_set]:
                batch_timing = []
                if i_step >= self.cfg.epoch_size:
                    break

                # read data to device
                img1, img2 = data["img1"].to(self.device), data["img2"].to(self.device)
                full_seg1, full_seg2 = data["full_seg1"].to(self.device), data[
                    "full_seg2"
                ].to(self.device)

                # adj_map1, adj_map2 = data["adj_map1"].to(self.device), data[
                #     "adj_map2"
                # ].to(self.device)

                # timing: 1_data_loading
                batch_timing.append(time.time() - last_time)
                last_time = time.time()

                # run 1st pass
                res_dict = self.model(img1, img2, full_seg1, full_seg2, with_bk=True)

                # timing: 2_main_forward
                batch_timing.append(time.time() - last_time)
                last_time = time.time()

                flows_12, flows_21 = res_dict["flows_12"], res_dict["flows_21"]
                flows = [
                    torch.cat([flow12, flow21], 1)
                    for flow12, flow21 in zip(flows_12, flows_21)
                ]

                (
                    loss,
                    l_ph,
                    l_sm,
                    flow_mean,
                    flow_vis_mask12,
                    flow_vis_mask21,
                ) = self.loss_func(
                    flows, img1, img2, full_seg1=full_seg1, full_seg2=full_seg2
                )
                loss = loss.mean()
                l_ph = l_ph.mean()
                l_sm = l_sm.mean()
                flow_mean = flow_mean.mean()

                # timing: 3_main_loss
                batch_timing.append(time.time() - last_time)
                last_time = time.time()

                flow_ori = flows_12[0].detach()
                noc_ori = flow_vis_mask12.clone()  # non-occluded region

                ## ARFlow appearance/spatial transform
                if self.cfg.run_atst:
                    img1, img2 = data["img1_ph"].to(self.device), data["img2_ph"].to(
                        self.device
                    )
                    s = {
                        "imgs": [img1, img2],
                        "full_segs": [full_seg1, full_seg2],
                        "flows_f": [flow_ori],
                        "masks_f": [noc_ori],
                    }

                    st_res = (
                        self.sp_transform(deepcopy(s))
                        if self.cfg.run_st
                        else deepcopy(s)
                    )
                    flow_t, noc_t = st_res["flows_f"][0], st_res["masks_f"][0]

                    # run another pass
                    img1_st, img2_st = st_res["imgs"]
                    full_seg1_st, full_seg2_st = st_res["full_segs"]
                    # adj_map1_st, adj_map2_st = full_segs_to_adj_maps(
                    #     full_seg1_st
                    # ), full_segs_to_adj_maps(full_seg2_st)
                    flow_t_pred = self.model(
                        img1_st, img2_st, full_seg1_st, full_seg2_st, with_bk=False
                    )["flows_12"][0]

                    if not self.cfg.mask_st:
                        noc_t = torch.ones_like(noc_t)
                    l_atst = (
                        (flow_t_pred - flow_t).abs() + self.cfg.ar_eps
                    ) ** self.cfg.ar_q
                    l_atst = (l_atst * noc_t).mean() / (noc_t.mean() + 1e-7)

                    loss += self.cfg.w_ar * l_atst
                else:
                    l_atst = torch.zeros_like(loss)

                # timing: 4_atst
                batch_timing.append(time.time() - last_time)
                last_time = time.time()

                ## Occlusion augmentation + moving objs
                if self.cfg.run_ot:

                    img1_ot, img2_ot = data["img1_ph"].to(self.device), data[
                        "img2_ph"
                    ].to(self.device)
                    full_seg1_ot, full_seg2_ot = data["full_seg1"].to(
                        self.device
                    ), data["full_seg2"].to(self.device)

                    flow_ot = flow_ori.clone()
                    noc_ot = noc_ori.clone()

                    # add fake objects if applicable
                    if (
                        self.cfg.key_obj_aug
                        and self.obj_cache.count == self.obj_cache.cache_size
                    ):

                        B = img1_ot.shape[0]
                        out = self.obj_cache.pop(
                            B * self.cfg.key_obj_count, with_aug=True
                        )
                        all_obj_mask, all_img_src, all_mean_flow = out

                        for _round in range(self.cfg.key_obj_count):
                            obj_mask = all_obj_mask[_round * B : (_round + 1) * B]
                            img_src = all_img_src[_round * B : (_round + 1) * B]
                            mean_flow = all_mean_flow[_round * B : (_round + 1) * B]

                            input_dict = {
                                "img1_tgt": img1_ot,
                                "img2_tgt": img2_ot,
                                "full_seg1_st": full_seg1_ot,
                                "full_seg2_st": full_seg2_ot,
                                "flow_tgt": flow_ot,
                                "noc_tgt": noc_ot,
                                "img_src": img_src.to(self.device),
                                "obj_mask": obj_mask.to(self.device),
                                "motion": mean_flow.to(self.device),
                            }

                            output = add_fake_object(input_dict)
                            (
                                img1_ot,
                                img2_ot,
                                full_seg1_ot,
                                full_seg2_ot,
                                flow_ot,
                                noc_ot,
                                obj_mask2,
                            ) = output

                    # run 3rd pass
                    imgs = torch.cat([img1_ot, img2_ot], 1)
                    full_segs_ot = torch.cat([full_seg1_ot, full_seg2_ot], 1)

                    imgs, full_segs_ot, flow_ot, noc_ot = random_crop(
                        imgs, full_segs_ot, flow_ot, noc_ot, self.cfg.ot_size
                    )
                    img1_ot, img2_ot = imgs[:, :3], imgs[:, 3:6]
                    full_seg1_ot, full_seg2_ot = (
                        full_segs_ot[:, :1],
                        full_segs_ot[:, 1:],
                    )
                    # adj_map1_ot, adj_map2_ot = full_segs_to_adj_maps(
                    #     full_seg1_ot
                    # ), full_segs_to_adj_maps(full_seg2_ot)

                    res_dict_ot = self.model(
                        img1_ot, img2_ot, full_seg1_ot, full_seg2_ot, with_bk=False
                    )
                    flow_ot_pred = res_dict_ot["flows_12"][0]

                    l_ot = (
                        (flow_ot_pred - flow_ot).abs() + self.cfg.ar_eps
                    ) ** self.cfg.ar_q
                    l_ot = (l_ot * noc_ot).mean() / (noc_ot.mean() + 1e-7)

                    loss += self.cfg.w_ar * l_ot

                    # push current object mask into cache for future use
                    if self.cfg.key_obj_aug:
                        valid_idx = ~torch.isnan(data["key_obj_mask"][:, 0, 0, 0])

                        if valid_idx.sum() > 0:  # at least one valid object
                            obj_mask = data["key_obj_mask"][valid_idx]
                            img = data["img1_ph"][valid_idx]
                            mean_flow = (flow_ori[valid_idx].cpu() * obj_mask).mean(
                                dim=[2, 3]
                            ) / obj_mask.mean(dim=[2, 3])

                            self.obj_cache.push(obj_mask, img, mean_flow)

                else:
                    l_ot = torch.zeros_like(loss)

                # timing: 5_ot
                batch_timing.append(time.time() - last_time)
                last_time = time.time()

                """
                import imageio
                img1_show = (img1_ot.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                imageio.imwrite('_DEBUG_IMG1.png', img1_show.reshape((-1, 832, 3)))
                img2_show = (img2_ot.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                imageio.imwrite('_DEBUG_IMG2.png', img2_show.reshape((-1, 832, 3)))
                flow_show = flow_to_image(flow_ot.cpu().numpy().transpose(0, 2, 3, 1).reshape((-1, 832, 2)))
                imageio.imwrite('_DEBUG_FLOW.png', flow_show)
                noc_show = (noc_ot.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
                imageio.imwrite('_DEBUG_NOC.png', noc_show.reshape((-1, 832, 1)))
                """

                # update meters
                key_meters.update(
                    [
                        loss.item(),
                        l_ph.item(),
                        l_sm.item(),
                        l_atst.item(),
                        l_ot.item(),
                        flow_mean.item(),
                    ],
                    img1.shape[0],
                )

                # compute gradient and do optimization step
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # clip gradient norm and back prop
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()

                # timing: 6_bakward_update
                batch_timing.append(time.time() - last_time)
                last_time = time.time()

                timing_meters.update(batch_timing)

                if (self.i_iter + 1) % self.cfg.record_freq == 0:
                    key_meters_vals = [None for _ in range(self.world_size)]
                    torch.distributed.all_gather_object(key_meters_vals, key_meters.val)
                    key_meters_vals = np.array(key_meters_vals).mean(axis=0)

                    if self.rank == 0:
                        for v, name in zip(key_meters_vals, key_meter_names):
                            self.summary_writer.add_scalar(
                                "train:{}/".format(name_dataset) + name,
                                v,
                                self.i_iter + 1,
                            )
                        self.summary_writer.add_scalar(
                            "train:{}/learning_rate".format(name_dataset),
                            self.optimizer.param_groups[0]["lr"],
                            self.i_iter + 1,
                        )

                        for v, name in zip(timing_meters.avg, timing_meter_names):
                            self.summary_writer.add_scalar(
                                "timing_batch_avg/" + name, v, self.i_iter + 1
                            )
                        timing_meters.reset(i=len(timing_meter_names))

                if (self.i_iter + 1) % self.cfg.print_freq == 0:
                    if self.rank == 0:
                        istr = "{}:{:04d}/{:04d}".format(
                            self.i_epoch, i_step + 1, self.cfg.epoch_size
                        ) + " Info {}".format(key_meters)
                        self.log(istr)

                i_step += 1
                self.i_iter += 1
        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):

        self.model.eval()

        for i_set, loader in enumerate(self.valid_loaders):
            name_dataset = loader.dataset.name

            flow_error_names = ["EPE_all", "EPE_noc", "EPE_occ"]
            flow_error_meters = AverageMeter(i=len(flow_error_names))

            for i_step, data in enumerate(loader):

                if i_step >= self.cfg.valid_size:
                    break

                # compute output
                img1 = data["img1"].to(self.device)
                img2 = data["img2"].to(self.device)
                full_seg1, full_seg2 = data["full_seg1"].to(self.device), data[
                    "full_seg2"
                ].to(self.device)

                # adj_map1, adj_map2 = data["adj_map1"].to(self.device), data[
                #     "adj_map2"
                # ].to(self.device)
                flow_gt = data["flow_gt"].numpy()
                occ_mask = data["occ_mask"].numpy()

                gt_flows = np.concatenate(
                    [flow_gt, np.ones_like(occ_mask), 1 - occ_mask], axis=3
                )

                res_dict = self.model(img1, img2, full_seg1, full_seg2, with_bk=False)
                flows = res_dict["flows_12"]
                pred_flows = flows[0].detach().cpu().numpy().transpose([0, 2, 3, 1])

                # evaluate
                es = evaluate_flow(gt_flows, pred_flows)
                flow_error_meters.update(es[:3], img1.shape[0])

            # write error to tf board.
            for value, name in zip(flow_error_meters.avg, flow_error_names):
                self.summary_writer.add_scalar(
                    "valid{}:{}/".format(i_set, name_dataset) + name,
                    value,
                    self.i_iter,
                )

        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            self.save_model(name="model")

        if self.i_epoch % 50 == 0:
            self.save_model(name="model_ep{}".format(self.i_epoch))

        return
