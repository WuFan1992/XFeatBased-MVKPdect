"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
 
 python -m modules.training.train --training_type vudnet_megadepth  --megadepth_root_path datasets/  --ckpt_save_path checkpoints/iter1 --batch_size 1 --n_steps 10000                           
 
 
"""

import argparse
import os
import time
import sys

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
    parser.add_argument('--save_ckpt_every', type=int, default=500,
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

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")] )
            
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


    def train(self):

        self.net.train()

        difficulty = 0.10
        subset_views_list = [5]
        desc_weights = {5: 1.0}

        p1s, p2s, H1, H2 = None, None, None, None
        d = None


        if self.augmentor is not None: # 使用COCO 合成的图片
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)
        
        if self.data_iter is not None: # 使用Megadepth 的图片
            d = next(self.data_iter)

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if not self.dry_run:
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
                valid_pairs = 0

                for b in range(self.batch_size):
                    
                    sample_data = self._get_sample_data(d, b)
                    sample_images = sample_data['images']
                    H_orig, W_orig = sample_images[0].shape[1:]
                    
                    # ===== 生成非互斥多视图子集（全量pair reuse） =====
                    batch_points_dict, id_to_idx = generate_multiview_subsets_noexclude(sample_data)
                    
                    # ====== forward 每个view =====
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
                        
                    # ===== 只使用 k=5 的5-view点，按10对独立生成对应点 =====
                    view_ids, _ = batch_points_dict[5]
                    if len(view_ids) < 5:
                        continue
                    
                    for vi_local in range(len(view_ids)):
                        for vj_local in range(vi_local + 1, len(view_ids)):
                            vi_sample_idx = id_to_idx[view_ids[vi_local]]
                            vj_sample_idx = id_to_idx[view_ids[vj_local]]
                            
                            try:
                                data_pair = {
                                    'images': [sample_data['images'][vi_sample_idx], sample_data['images'][vj_sample_idx]],
                                    'depths': [sample_data['depths'][vi_sample_idx], sample_data['depths'][vj_sample_idx]],
                                    'Ks': [sample_data['Ks'][vi_sample_idx], sample_data['Ks'][vj_sample_idx]],
                                    'T_0to': [torch.eye(3, 4, device=self.dev), 
                                             sample_data['T_0to'][vj_sample_idx] if 'T_0to' in sample_data else torch.eye(3, 4, device=self.dev)],
                                    'scales': [sample_data['scales'][vi_sample_idx] if 'scales' in sample_data else torch.tensor([1.0, 1.0]),
                                              sample_data['scales'][vj_sample_idx] if 'scales' in sample_data else torch.tensor([1.0, 1.0])],
                                }
                                if 'image_masks' in sample_data:
                                    data_pair['image_masks'] = [sample_data['image_masks'][vi_sample_idx], 
                                                               sample_data['image_masks'][vj_sample_idx]]
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
                            
                            feat_i = sample_map_at_coords(feats[vi_sample_idx], coords_i, H_orig, W_orig)
                            feat_j = sample_map_at_coords(feats[vj_sample_idx], coords_j, H_orig, W_orig)
                            
                            hmap_i = sample_map_at_coords(hmap[vi_sample_idx], coords_i, H_orig, W_orig)
                            hmap_j = sample_map_at_coords(hmap[vj_sample_idx], coords_j, H_orig, W_orig)
                            
                            mask_valid = vis_pair[:, 0] & vis_pair[:, 1]
                            if mask_valid.sum() < 10:
                                continue
                            
                            feat_i = feat_i[mask_valid]
                            feat_j = feat_j[mask_valid]
                            hmap_i = hmap_i[mask_valid]
                            hmap_j = hmap_j[mask_valid]
                            
                            loss_desc_pair, conf_pair = dual_softmax_loss(feat_i, feat_j, temp=0.2)
                            loss_desc_total += loss_desc_pair
                            loss_hmap_pair = keypoint_loss(hmap_i, conf_pair) + keypoint_loss(hmap_j, conf_pair)
                            loss_hmap_total += loss_hmap_pair
                            valid_pairs += 1
                        
                    loss_kpts_b = torch.zeros([], device=self.dev)
                    cnt = 0.0
                    for view_id in view_ids:
                        view_sample_idx = id_to_idx[view_id]
                        pred_hm = kpts[view_sample_idx]
                        img_v = sample_images[view_sample_idx]
                        loss_hm_v, acc_hm_v = alike_distill_loss(pred_hm[0], img_v)
                        loss_kpts_b += loss_hm_v
                        cnt += 1
                    loss_kpts_total += loss_kpts_b / max(cnt, 1)

                loss_desc = loss_desc_total / max(valid_pairs, 1)
                loss_hmap = loss_hmap_total / max(valid_pairs, 1)
                loss_kpts = loss_kpts_total / max(1, self.batch_size)
            
                # ===== 总 loss =====
                loss = loss_desc + 1.0 * loss_hmap + 1.0 * loss_kpts
                
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
                pbar.set_description( 'Loss: {:.4f}  loss_c: {:.3f}  loss_hm: {:.3f}  loss_kp: {:.3f}  pairs: {} '.format(
                                                                        loss.item(), loss_desc, loss_hmap, loss_kpts, valid_pairs) )
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Loss/coarse', loss_desc, i)
                self.writer.add_scalar('Loss/reliability', loss_hmap, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kpts, i)
                self.writer.add_scalar('Metric/valid_pairs', valid_pairs, i)



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
