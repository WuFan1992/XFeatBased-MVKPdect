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
        subset_views_list = [5, 4, 3, 2]
        desc_weights = {5: 1.0, 4: 1.0, 3: 1.0, 2: 1.0}

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

                
                loss_desc_total, valid_batch = 0.0, 0
                loss_hmap_total = 0.0
                loss_kpts_total = 0.0

                for b in range(self.batch_size):
                    
                    sample_data = self._get_sample_data(d, b)
                    sample_images = sample_data['images']
                    H_orig, W_orig = sample_images[0].shape[1:]
                    
                    # ===== 生成互斥子集 =====
                    batch_points_dict, id_to_idx = generate_exclusive_subsets(sample_data)
                    
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
                        
                        # ==== 对每个view 的子集计算loss =====
                    for k in subset_views_list:
                        subset_ids, (corrs_k, vis_k) = batch_points_dict[k]
                        N_points = corrs_k.shape[0]

                        # ==== 限制最小点数 =====
                        if N_points < 10:
                            continue
                            
                        # ===== 限制最大点数 =====
                        max_points = 5000
                        if N_points > max_points:
                            idx = torch.randperm(N_points)[:max_points]
                            corrs_k = corrs_k[idx]
                            vis_k = vis_k[idx]
                                
                        # ===== 采样 multi-view 特征 =====
                        feat_per_point, hmap_per_point, kpts_per_point = [], [], []
                        for local_v, global_v in enumerate(subset_ids):
                            coords = corrs_k[:, local_v, :].to(self.dev)
                            # 采样描述子

                            feat_sample = sample_map_at_coords(feats[local_v], coords, H_orig, W_orig)
                            hmap_sample = sample_map_at_coords(hmap[local_v], coords, H_orig, W_orig)
                            kpts_sample = sample_map_at_coords(kpts[local_v], coords, H_orig, W_orig) 
                                
                            feat_per_point.append(feat_sample)
                            hmap_per_point.append(hmap_sample)
                            kpts_per_point.append(kpts_sample)
                                
                        # stack → [N_points, k, C]
                        stacked_feats = torch.stack(feat_per_point, dim=1)
                        stacked_hmaps = torch.stack(hmap_per_point, dim=1)
                        stacked_kpts = torch.stack(kpts_per_point, dim=1)
                            
                        # ===== visibility mask =====
                        visibility = torch.ones((N_points, k), device=self.dev, dtype=torch.bool) if vis_k is None else vis_k.bool().to(self.dev)
                            
                        # ===== 计算 multi-view loss =====
                        loss_desc_b = loss = mv_infonce_masked(stacked_feats, visibility, tau=0.2)
                        loss_desc_total += desc_weights[k] * loss_desc_b
                            
                        # ===== 计算 reliability loss =====
                        loss_hmap_b = multi_view_xfeat_heatmap_loss(
                                    stacked_hmaps,
                                    stacked_kpts,
                                    visibility,
                                    tau=0.2
                        )
                        loss_hmap_total += desc_weights[k] * loss_hmap_b
                            
                        # ===== kpts loss =====
                        loss_kpts_b = 0.0
                        for v in range(k):
                            pred_hm = kpts[v]  # [1, 65, H/8, W/8]
                            img_v = d['images'][v]  # [3, H_orig, W_orig] or [1,...]
                            loss_hm_v, acc_hm_v = alike_distill_loss(pred_hm[0], img_v[0])
                            loss_kpts_b += loss_hm_v

                        loss_kpts_total += loss_kpts_b / k
                        valid_batch += 1

                loss_desc = loss_desc_total / valid_batch if valid_batch > 0 else torch.tensor(0.0, device=self.dev, requires_grad=True)
                loss_hmap = loss_hmap_total / valid_batch if valid_batch > 0 else torch.tensor(0.0, device=self.dev, requires_grad=True)
                loss_kpts = loss_kpts_total / valid_batch if valid_batch > 0 else torch.tensor(0.0, device=self.dev, requires_grad=True)
            

                # ===== 总 loss =====
                loss = (
                loss_desc
                + 1.0 * loss_hmap      # 增加 reliability 权重，更接近 XFeat supervision
                + 1.0 * loss_kpts
                )
                # Compute Backward Pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i+1) % self.save_ckpt_every == 0:
                    print('saving iter ', i+1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/{self.model_name}_{i+1}.pth')
                pbar.set_description( 'Loss: {:.4f}  loss_c: {:.3f}  loss_kp: {:.3f}  loss_kp_pos: {:.3f} '.format(
                                                                        loss.item(),  loss_desc, loss_hmap, loss_kpts) )
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Loss/coarse', loss_desc, i)
                #self.writer.add_scalar('Loss/fine', loss_coord, i)
                self.writer.add_scalar('Loss/reliability', loss_hmap, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kpts, i)



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
