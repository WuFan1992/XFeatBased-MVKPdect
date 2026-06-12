"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
 
 python -m modules.training.train --training_type vudnet_megadepth  --megadepth_root_path datasets/  --ckpt_save_path checkpoints/iter1 --batch_size 1 --n_steps 10000                           
 
 
"""

import argparse
import os
import time
import sys
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="VUDNet training script.")

    parser.add_argument('--megadepth_root_path', type=str, default='/ssd/guipotje/Data/MegaDepth',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='/homeLocal/guipotje/sshfs/datasets/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='vudnet_default',
                        choices=['vudnet_default', 'vudnet_synthetic', 'vudnet_megadepth'],
                        help='Training scheme. vudnet_default uses both megadepth & synthetic warps.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=10000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height. Default is (800, 608).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=5000,
                        help='Save checkpoints every N steps. Default is 500.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args

args = parse_arguments()

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from modules.vudnetmodel import VUDNetModel
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *

from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth.megadepth_warper import *
from modules.dataset.megadepth.utils import *
from torch.utils.data import Dataset, DataLoader

def plot_multi_view_matches(images, multi_corrs, vis=None, max_points=50, save_path=None):
    """
    可视化多视图匹配点轨迹（严格可见性过滤）
    
    Args:
        images: list of images [C,H,W] 或 [H,W] 或 torch.Tensor
        multi_corrs: [N, V, 2] tensor 或 numpy array
        vis: [N, V] bool，可见性掩码，如果 None，则默认全部可见
        max_points: 最大显示点数
        save_path: 保存路径，如果 None 则 plt.show()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # ===== 转 numpy image =====
    def to_numpy_image(img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 4:  # [B,C,H,W]
            img = img[0]
        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img[0]
            elif img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        return img

    imgs = [to_numpy_image(img) for img in images]
    V = len(imgs)

    H_list = [img.shape[0] for img in imgs]
    W_list = [img.shape[1] for img in imgs]

    H = max(H_list)
    W = sum(W_list)

    canvas = np.zeros((H, W, 3), dtype=imgs[0].dtype)
    offsets = []
    cur_w = 0
    for img in imgs:
        h, w = img.shape[:2]
        canvas[:h, cur_w:cur_w+w] = img
        offsets.append(cur_w)
        cur_w += w

    # ===== 转 numpy 数据 =====
    if isinstance(multi_corrs, torch.Tensor):
        multi_corrs = multi_corrs.detach().cpu().numpy()
    if vis is None:
        vis = np.ones((multi_corrs.shape[0], V), dtype=bool)
    elif isinstance(vis, torch.Tensor):
        vis = vis.detach().cpu().numpy()

    N = multi_corrs.shape[0]
    if N == 0:
        print("No correspondences")
        return

    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        multi_corrs = multi_corrs[idx]
        vis = vis[idx]
        N = max_points

    # ===== 绘制 =====
    plt.figure(figsize=(15, 5))
    plt.imshow(canvas)
    colors = plt.cm.jet(np.linspace(0, 1, N))

    for i in range(N):
        # 检查该点在所有视图中是否都可见且在图像内
        keep_track = True
        for j in range(V):
            x, y = multi_corrs[i, j]
            h, w = imgs[j].shape[:2]
            if not vis[i, j] or x < 0 or x >= w or y < 0 or y >= h:
                keep_track = False
                break

        if not keep_track:
            continue  # 只保留完全可见的轨迹

        # 画点和连线
        prev_pt = None
        for j in range(V):
            x, y = multi_corrs[i, j]
            plt.scatter(x + offsets[j], y, c=[colors[i]], s=20)
            if prev_pt is not None:
                x0, y0, j0 = prev_pt
                plt.plot([x0 + offsets[j0], x + offsets[j]], [y0, y], c=colors[i], linewidth=1)
            prev_pt = (x, y, j)

    plt.axis('off')
    plt.title(f"Multi-view Correspondences ({V} views)")
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=True)

def visualize_multi_view_matches(batch_data, batch_points_dict, id_to_idx):
    """
    可视化多视图匹配子集
    """
    images_vis = []
    for img in batch_data['images']:
        if isinstance(img, torch.Tensor) and img.dim() == 4:
            images_vis.append(img[0])
        else:
            images_vis.append(img)

    for k in sorted(batch_points_dict.keys(), reverse=True):
        subset_ids, (corrs_k, vis_k) = batch_points_dict[k]
        if corrs_k.shape[0] == 0:
            print(f"[VIS] No {k}-view points")
            continue

        # 图像顺序和 subset 对齐
        imgs_k = [images_vis[id_to_idx[i]] for i in subset_ids]

        if vis_k is not None:
            vis_k = vis_k.astype(bool) if isinstance(vis_k, np.ndarray) else vis_k.bool()

        plot_multi_view_matches(
            images=imgs_k,
            multi_corrs=corrs_k,
            vis=vis_k,
            max_points=50,
            save_path=None
        )


class Trainer():
    """
        Class for training VUDNet with default params as described in the paper.
        We use a blend of MegaDepth (labeled) pairs with synthetically warped images (self-supervised).
        The major bottleneck is to keep loading huge megadepth h5 files from disk, 
        the network training itself is quite fast.
    """

    def __init__(self, megadepth_root_path, 
                       synthetic_root_path, 
                       ckpt_save_path, 
                       model_name = 'vudnet_default',
                       batch_size = 10, n_steps = 160_000, lr= 3e-4, gamma_steplr=0.5, 
                       training_res = (800, 608), device_num="0", dry_run = False,
                       save_ckpt_every = 5000):

        self.dev = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = VUDNetModel().to(self.dev)

        #Setup optimizer 
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()) , lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        ##################### Synthetic COCO INIT ##########################
        # 从COCO 读取图片，然后做随机的几何变换，光照变换和裁剪缩放，从而得到成对的图像和几何变换矩阵
        if model_name in ('vudnet_default', 'vudnet_synthetic'):
            self.augmentor = AugmentationPipe(
                                        img_dir = synthetic_root_path,
                                        device = self.dev, load_dataset = True,
                                        batch_size = int(self.batch_size * 0.4 if model_name=='vudnet_default' else batch_size), # 采用40% synthetic 数据 60% Megadepth 数据
                                        out_resolution = training_res, 
                                        warp_resolution = training_res,
                                        sides_crop = 0.1,
                                        max_num_imgs = 3_000,
                                        num_test_imgs = 5,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 4_000
                                        )
        else:
            self.augmentor = None
        ##################### Synthetic COCO END #######################


        ##################### MEGADEPTH INIT ##########################
        if model_name in ('vudnet_default', 'vudnet_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"
            
            #TRAIN_BASE_PATH = f"{megadepth_root_path}"
            #TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/thirtysceneData"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            
            scene_groups = defaultdict(list)
            for path in npz_paths:
                filename = os.path.basename(path)
                scene_id = filename.split('_')[0]
                scene_groups[scene_id].append(path)
                
            datasets = []
            for scene_id, scene_paths in scene_groups.items():
                datasets.append(
                    MegaDepthDataset(
                    root_dir=TRAINVAL_DATA_SOURCE,
                    npz_paths=scene_paths
                ))
            
            data = torch.utils.data.ConcatDataset(datasets)
            
            # 一个bacth 里面有10 对匹配图像
            self.data_loader = DataLoader(data, 
                                          batch_size=int(self.batch_size * 0.6 if model_name=='vudnet_default' else batch_size),
                                          shuffle=True)
            self.data_iter = iter(self.data_loader)

        else:
            self.data_iter = None
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name
        
    def _get_sample_data(self, batch_data, b):
        sample_data = {}
        for key, value in batch_data.items():
            if isinstance(value, list):
                if len(value) == 0:
                    sample_data[key] = []
                elif isinstance(value[0], torch.Tensor):
                    sample_data[key] = [v[b] for v in value]
                else:
                    sample_data[key] = value[b]
            elif isinstance(value, torch.Tensor):
                sample_data[key] = value[b]
            else:
                try:
                    sample_data[key] = value[b]
                except Exception:
                    sample_data[key] = value
        return sample_data

    def _sample_hard_negatives_from_multiview(self, feat_i, feat_j, feats, sample_images, 
                                               sample_data, H_orig, W_orig, coords_i, coords_j,
                                               num_neg_per_point=4, num_candidates=16, radius=24):
        """
        Sample geometry-aware hard negatives from other views.

        We project matched points from view 0 into additional views and mine negatives
        from a local neighbourhood around those projected locations. This is much harder
        than uniformly sampling random points across the image.

        Args:
            feat_i, feat_j: [M, C] matched features from view 0 and 1
            feats: list of feature maps for all views
            sample_images: list of images for all views
            sample_data: metadata for all views
            H_orig, W_orig: original image dimensions
            coords_i, coords_j: [M, 2] matched pixel coordinates in views 0 and 1
            num_neg_per_point: number of hard negatives to return per matched point
            num_candidates: number of candidate negatives to mine per point
            radius: spatial radius (pixels) around the projected location to search

        Returns:
            hard_neg_feat_i: [M, K, C] hard negative features for view 0
            hard_neg_feat_j: [M, K, C] hard negative features for view 1
        """
        M = feat_i.shape[0]
        C = feat_i.shape[1]
        V = len(feats)

        if V < 3 or M == 0:
            return None, None

        hard_neg_feat_i_list = []
        hard_neg_feat_j_list = []

        # Use views 2, 3, 4 as sources for hard negatives
        neg_views = list(range(2, min(V, 5)))
        if len(neg_views) == 0:
            return None, None

        # Pre-flatten coords_i for later candidate generation
        coords_i = coords_i.detach()

        for neg_view_idx in neg_views:
            neg_feat = feats[neg_view_idx]  # [1, C, H, W]

            # Project view-0 matched points into the candidate view
            try:
                depth0 = sample_data['depths'][0]
                depth_v = sample_data['depths'][neg_view_idx]
                K0 = sample_data['Ks'][0]
                Kv = sample_data['Ks'][neg_view_idx]
                T_0to_v = sample_data['T_0to'][neg_view_idx]

                coords0_b = coords_i.unsqueeze(0)
                depth0_b = depth0.unsqueeze(0) if depth0.dim() == 2 else depth0
                depth_v_b = depth_v.unsqueeze(0) if depth_v.dim() == 2 else depth_v
                valid_mask, proj_coords = warp_kpts(
                    coords0_b,
                    depth0_b,
                    depth_v_b,
                    T_0to_v.unsqueeze(0),
                    K0.unsqueeze(0),
                    Kv.unsqueeze(0)
                )
                valid_mask = valid_mask[0]
                proj_coords = proj_coords[0]
                proj_coords[:, 0].clamp_(0, W_orig - 1)
                proj_coords[:, 1].clamp_(0, H_orig - 1)
            except Exception:
                proj_coords = coords_i
                valid_mask = torch.ones((M,), dtype=torch.bool, device=self.dev)

            if valid_mask.sum() == 0:
                continue

            # Local randomized candidates around the projected location
            offsets = (torch.rand((M, num_candidates, 2), device=self.dev) * 2 - 1) * radius
            min_dist = 4.0
            if num_candidates > 1:
                too_close = offsets.abs() < min_dist
                offsets = offsets + torch.sign(offsets) * min_dist * too_close.float()
            cand_coords = proj_coords.unsqueeze(1) + offsets
            cand_coords[..., 0].clamp_(0, W_orig - 1)
            cand_coords[..., 1].clamp_(0, H_orig - 1)
            cand_coords = cand_coords.view(-1, 2)

            cand_feats = sample_map_at_coords(neg_feat, cand_coords, H_orig, W_orig)
            if cand_feats.numel() == 0:
                continue

            cand_feats = cand_feats.view(M, -1, C)
            k = min(num_neg_per_point, cand_feats.shape[1])

            sim_i = torch.einsum('mc,mkc->mk', feat_i, cand_feats)
            sim_j = torch.einsum('mc,mkc->mk', feat_j, cand_feats)

            _, idx_i = sim_i.topk(k, dim=1, largest=True, sorted=False)
            _, idx_j = sim_j.topk(k, dim=1, largest=True, sorted=False)
            idx_i = idx_i.unsqueeze(-1).expand(-1, -1, C)
            idx_j = idx_j.unsqueeze(-1).expand(-1, -1, C)

            hard_neg_feat_i_list.append(torch.gather(cand_feats, 1, idx_i))
            hard_neg_feat_j_list.append(torch.gather(cand_feats, 1, idx_j))

        if len(hard_neg_feat_i_list) == 0:
            return None, None

        hard_neg_feat_i = torch.cat(hard_neg_feat_i_list, dim=1)
        hard_neg_feat_j = torch.cat(hard_neg_feat_j_list, dim=1)

        return hard_neg_feat_i, hard_neg_feat_j


    def train(self):

        self.net.train()

        difficulty = 0.10

        p1s, p2s, H1, H2 = None, None, None, None
        d = None


        if self.augmentor is not None: # 使用COCO 合成的图片
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)
        
        if self.data_iter is not None: # 使用Megadepth 的图片
            d = next(self.data_iter)

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):

                if self.data_iter is not None:
                    try:
                        # Get the next MD batch
                        d = next(self.data_iter)

                    except StopIteration:
                        print("End of DATASET!")
                        # If StopIteration is raised, create a new iterator.
                        self.data_iter = iter(self.data_loader)
                        d = next(self.data_iter)
                
                loss_desc_total = torch.zeros([], device=self.dev)
                loss_hmap_total = torch.zeros([], device=self.dev)
                loss_kpts_total = torch.zeros([], device=self.dev)
                loss_var_total = torch.zeros([], device=self.dev)
                valid_pairs = 0
                valid_var_subsets = 0

                for b in range(self.batch_size):
                    
                    sample_data = self._get_sample_data(d, b)
                    sample_images = sample_data['images']
                    H_orig, W_orig = sample_images[0].shape[1:]
                    
                    # Forward each view for this sample
                    V = len(sample_images)
                    feats, kpts, hmap, vars = [], [], [], []
                    for v in range(V):
                        img = sample_images[v]
                        if img.dim() == 3:
                            img = img.unsqueeze(0)
                        feat_v, kpt_v, hmap_v, var_v = self.net(img.to(self.dev))
                        feats.append(feat_v)
                        kpts.append(kpt_v)
                        hmap.append(hmap_v)
                        vars.append(var_v)

                    # Only use a single pair for training: view 0 and view 1
                    try:
                        data_pair = {
                            'images': [sample_data['images'][0], sample_data['images'][1]],
                            'depths': [sample_data['depths'][0], sample_data['depths'][1]],
                            'Ks': [sample_data['Ks'][0], sample_data['Ks'][1]],
                            'T_0to': [torch.eye(3, 4, device=self.dev), sample_data['T_0to'][1]],
                            'scales': [sample_data['scales'][0], sample_data['scales'][1]],
                        }
                        if 'image_masks' in sample_data:
                            data_pair['image_masks'] = [sample_data['image_masks'][0], sample_data['image_masks'][1]]
                    except (KeyError, IndexError, TypeError):
                        continue

                    corrs_pair, vis_pair = generate_pairwise_corrs_independent(
                        data_pair,
                        view_idx0=0,
                        view_idx1=1,
                        scale=8
                    )

                    N_pair = corrs_pair.shape[0]
                    if N_pair < 10:
                        continue

                    max_points = 5000
                    if N_pair > max_points:
                        idx = torch.randperm(N_pair)[:max_points]
                        corrs_pair = corrs_pair[idx]
                        vis_pair = vis_pair[idx]

                    corrs_pair = corrs_pair.to(self.dev)
                    vis_pair = vis_pair.to(self.dev)

                    coords_i = corrs_pair[:, 0, :].to(self.dev)
                    coords_j = corrs_pair[:, 1, :].to(self.dev)

                    feat_i = sample_map_at_coords(feats[0], coords_i, H_orig, W_orig)
                    feat_j = sample_map_at_coords(feats[1], coords_j, H_orig, W_orig)

                    hmap_i = sample_map_at_coords(hmap[0], coords_i, H_orig, W_orig)
                    hmap_j = sample_map_at_coords(hmap[1], coords_j, H_orig, W_orig)

                    mask_valid = vis_pair[:, 0] & vis_pair[:, 1]
                    if mask_valid.sum() < 10:
                        continue

                    feat_i = feat_i[mask_valid]
                    feat_j = feat_j[mask_valid]
                    hmap_i = hmap_i[mask_valid]
                    hmap_j = hmap_j[mask_valid]

                    # ===== Extract hard negatives from multi-view subsets =====
                    hard_neg_feat_i = None
                    hard_neg_feat_j = None
                    
                    if V >= 3:  # Only if we have 3+ views
                        hard_neg_feat_i, hard_neg_feat_j = self._sample_hard_negatives_from_multiview(
                            feat_i, feat_j, feats, sample_images,
                            sample_data, H_orig, W_orig,
                            coords_i, coords_j,
                            num_neg_per_point=4,
                            num_candidates=16,
                            radius=24
                        )
                    
                    loss_desc_pair, conf_pair = dual_softmax_loss(
                        feat_i, feat_j, 
                        temp=0.2,
                        hard_neg_X=hard_neg_feat_i,
                        hard_neg_Y=hard_neg_feat_j,
                        hard_neg_weight=0.3,
                        margin=0.1
                    )
                    loss_desc_total += loss_desc_pair
                    loss_hmap_pair = keypoint_loss(hmap_i, conf_pair) + keypoint_loss(hmap_j, conf_pair)
                    loss_hmap_total += loss_hmap_pair
                    valid_pairs += 1
                    
                    loss_kpts_b = torch.zeros([], device=self.dev)
                    for view_sample_idx in [0, 1]:
                        pred_hm = kpts[view_sample_idx]
                        img_v = sample_images[view_sample_idx]
                        loss_hm_v, acc_hm_v = alike_distill_loss(pred_hm[0], img_v)
                        loss_kpts_b += loss_hm_v
                    loss_kpts_total += loss_kpts_b / 2.0

                    # ===== Variance Loss from Multi-View Subsets =====
                    batch_points_dict, id_to_idx = generate_exclusive_subsets(sample_data)
                    #visualize_multi_view_matches(sample_data, batch_points_dict, id_to_idx)
                    subset_views_list = [5, 4, 3, 2]
                    
                    for k in subset_views_list:
                        if k not in batch_points_dict:
                            continue
                        
                        subset_ids, (corrs_k, vis_k) = batch_points_dict[k]
                        if subset_ids is None or len(subset_ids) < k:
                            continue
                        
                        N_points = corrs_k.shape[0] if corrs_k is not None else 0
                        
                        if N_points < 20:
                            continue
                        
                        # Limit maximum points
                        max_points = 5000
                        if N_points > max_points:
                            idx = torch.randperm(N_points)[:max_points]
                            corrs_k = corrs_k[idx]
                            vis_k = vis_k[idx]
                        
                        corrs_k = corrs_k.to(self.dev)
                        vis_k = vis_k.to(self.dev)
                        
                        # Sample features and variance from k views
                        f_inv_per_point, sigma_per_point = [], []
                        
                        for v_local in range(k):
                            view_id = subset_ids[v_local]
                            view_idx = id_to_idx[view_id]
                            coords = corrs_k[:, v_local, :].to(self.dev)
                            
                            f_inv_sample = sample_map_at_coords(feats[view_idx], coords, H_orig, W_orig)
                            sigma_sample = sample_map_at_coords(vars[view_idx], coords, H_orig, W_orig)
                            
                            f_inv_per_point.append(f_inv_sample)
                            sigma_per_point.append(sigma_sample)
                        
                        f_inv_k = torch.stack(f_inv_per_point, dim=1)  # [N, k, C]
                        sigma_k = torch.stack(sigma_per_point, dim=1)  # [N, k, 1]
                        
                        visibility = vis_k.bool() if vis_k is not None else torch.ones((N_points, k), dtype=torch.bool, device=self.dev)
                        
                        # Compute sigma loss
                        loss_var_k, var_target = sigma_consistency_loss(f_inv_k, sigma_k, visibility)
                        loss_var_total += loss_var_k
                        valid_var_subsets += 1

                loss_desc = loss_desc_total / max(valid_pairs, 1)
                loss_hmap = loss_hmap_total / max(valid_pairs, 1)
                loss_kpts = loss_kpts_total / max(1, self.batch_size)
                loss_var = loss_var_total / max(valid_var_subsets, 1)
            
                # ===== Total loss =====
                loss = loss_desc + 1.0 * loss_hmap + 1.0 * loss_kpts + 1.0 * loss_var
                
                if valid_pairs == 0:
                    print(f"[WARN] Iter {i}: no valid pairs")
                    continue

                if not loss.requires_grad:
                    print(f"[WARN] Iter {i}: loss has no grad")
                    continue
                
                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i+1}.pth')
                pbar.set_description( 'Loss: {:.4f}  loss_c: {:.3f}  loss_hm: {:.3f}  loss_kp: {:.3f}  loss_var: {:.3f}  pairs: {}  var_subsets: {} '.format(
                                                                        loss.item(), loss_desc, loss_hmap, loss_kpts, loss_var, valid_pairs, valid_var_subsets) )
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Loss/coarse', loss_desc, i)
                self.writer.add_scalar('Loss/reliability', loss_hmap, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kpts, i)
                self.writer.add_scalar('Loss/variance', loss_var, i)
                self.writer.add_scalar('Metric/valid_pairs', valid_pairs, i)
                self.writer.add_scalar('Metric/valid_var_subsets', valid_var_subsets, i)



if __name__ == '__main__':

    trainer = Trainer(
        megadepth_root_path=args.megadepth_root_path, 
        synthetic_root_path=args.synthetic_root_path, 
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every
    )

    #The most fun part
    trainer.train()
