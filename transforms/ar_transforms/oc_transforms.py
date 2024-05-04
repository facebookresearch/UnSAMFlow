# from skimage.color import rgb2yuv
# import cv2
import numpy as np
import torch

# from fast_slic.avx2 import SlicAvx2 as Slic
# from skimage.segmentation import slic as sk_slic

from utils.warp_utils import flow_warp


# def run_slic_pt(img_batch, n_seg=200, compact=10, rd_select=(8, 16), fast=True):  # Nx1xHxW
#     """

#     :param img: Nx3xHxW 0~1 float32
#     :param n_seg:
#     :param compact:
#     :return: Nx1xHxW float32
#     """
#     B = img_batch.size(0)
#     dtype = img_batch.type()
#     img_batch = np.split(
#         img_batch.detach().cpu().numpy().transpose([0, 2, 3, 1]), B, axis=0)
#     out = []
#     if fast:
#         fast_slic = Slic(num_components=n_seg, compactness=compact, min_size_factor=0.8)
#     for img in img_batch:
#         img = np.copy((img * 255).squeeze(0).astype(np.uint8), order='C')
#         if fast:
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#             seg = fast_slic.iterate(img)
#         else:
#             seg = sk_slic(img, n_segments=200, compactness=10)

#         if rd_select is not None:
#             n_select = np.random.randint(rd_select[0], rd_select[1])
#             select_list = np.random.choice(range(0, np.max(seg) + 1), n_select,
#                                            replace=False)

#             seg = np.bitwise_or.reduce([seg == seg_id for seg_id in select_list])
#         out.append(seg)
#     x_out = torch.tensor(np.stack(out)).type(dtype).unsqueeze(1)
#     return x_out


def random_crop(img, full_segs, flow, occ_mask, crop_sz):
    """

    :param img: Nx6xHxW
    :param flows: n * [Nx2xHxW]
    :param occ_masks: n * [Nx1xHxW]
    :param crop_sz:
    :return:
    """
    _, _, h, w = img.size()
    c_h, c_w = crop_sz

    if c_h == h and c_w == w:
        return img, flow, occ_mask

    x1 = np.random.randint(0, w - c_w)
    y1 = np.random.randint(0, h - c_h)
    img = img[:, :, y1 : y1 + c_h, x1 : x1 + c_w]
    full_segs = full_segs[:, :, y1 : y1 + c_h, x1 : x1 + c_w]
    flow = flow[:, :, y1 : y1 + c_h, x1 : x1 + c_w]
    occ_mask = occ_mask[:, :, y1 : y1 + c_h, x1 : x1 + c_w]

    return img, full_segs, flow, occ_mask


# def semantic_connected_components(
#     semseg, class_indices, width_range=(30, 200), height_range=(30, 100)
# ):
#     """
#     Input:
#         semsegs: Onehot semantic segmentations of size [c, H, W]
#         class_indices: A list of the indices of the classes of interest.
#                        For example, [car_idx] or [sign_idx, pole_idx, traffic_light_idx]
#         width_range, height_range: The width and height ranges for the objects of interest.
#     Output:
#         list of masks for cars in the size range
#     """

#     curr_sem = semseg[class_indices].sum(dim=0)
#     curr_sem = (curr_sem[:, :, None] * 255).numpy().astype(np.uint8)
#     num_labels, labels = cv2.connectedComponents(curr_sem)

#     sem_list = []
#     # 0 is background, so ignore them.
#     for i in range(1, num_labels):
#         curr_obj = labels == i
#         hs, ws = np.where(curr_obj)
#         h_len, w_len = np.max(hs) - np.min(hs), np.max(ws) - np.min(ws)
#         if (height_range[0] < h_len < height_range[1]) and (
#             width_range[0] < w_len < width_range[1]
#         ):
#             if (
#                 curr_obj.sum() / ((h_len + 1) * (w_len + 1)) > 0.6
#             ):  # filter some wrong car estimates or largely occluded cars
#                 sem_list.append(curr_obj)

#     return sem_list


# def find_semantic_group(semseg, class_indices, win_width=200):
#     curr_sem = semseg[class_indices].sum(dim=0).numpy()
#     freq = curr_sem.mean(axis=0)  # 1d frequency
#     freq_win = np.convolve(
#         freq, np.ones(win_width) / win_width, mode="valid"
#     )  # find the most frequent window

#     if max(freq_win) > 0.1:
#         # optimal window: [left, right)
#         left = np.argmax(freq_win)
#         right = left + win_width
#         curr_sem[:, :left] = 0
#         curr_sem[:, right:] = 0
#         return curr_sem

#     else:
#         return None


def add_fake_object(input_dict):

    # prepare input
    img1_ot = input_dict["img1_tgt"]
    img2_ot = input_dict["img2_tgt"]
    full_seg1_ot = input_dict["full_seg1_st"]
    full_seg2_ot = input_dict["full_seg2_st"]
    flow_ot = input_dict["flow_tgt"]
    noc_ot = input_dict["noc_tgt"]

    img = input_dict["img_src"]
    obj_mask = input_dict["obj_mask"]
    motion = input_dict["motion"][:, :, None, None]

    b, _, h, w = img1_ot.shape
    N1 = full_seg1_ot.max()
    N2 = full_seg2_ot.max()

    # add object to frame 1
    img1_ot = obj_mask * img + (1 - obj_mask) * img1_ot
    full_seg1_ot = obj_mask * (N1 + 1) + (1 - obj_mask) * full_seg1_ot

    # add object to frame 2
    new_obj_mask = flow_warp(obj_mask, -motion.repeat(1, 1, h, w), pad="zeros")
    new_img = flow_warp(img, -motion.repeat(1, 1, h, w), pad="border")
    img2_ot = new_obj_mask * new_img + (1 - new_obj_mask) * img2_ot
    full_seg2_ot = new_obj_mask * (N2 + 1) + (1 - new_obj_mask) * full_seg2_ot

    # change flow
    flow_ot = obj_mask * motion + (1 - obj_mask) * flow_ot
    noc_ot = torch.max(noc_ot, obj_mask)  # where we are confident about flow_ot

    return img1_ot, img2_ot, full_seg1_ot, full_seg2_ot, flow_ot, noc_ot, new_obj_mask
