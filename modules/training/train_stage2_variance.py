import argparse
import os
import time
import glob

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from modules.vudnetmodel import VUDNetModel
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *
from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth.megadepth_warper import *
from modules.dataset.megadepth.utils import *

from tqdm import tqdm




"""
python -m modules.training.train_stage2_variance \
  --megadepth_root_path datasets \
  --stage1_ckpt checkpoints/stage1/stage1_10000.pth \
  --ckpt_save_path checkpoints/stage2 \
  --batch_size 1 \
  --n_steps 10000 \
  --device_num 0
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Stage-2 variance and multi-view descriptor consistency training")
    parser.add_argument('--megadepth_root_path', type=str, required=True,
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--stage1_ckpt', type=str, required=True,
                        help='Path to the stage-1 checkpoint (.pth).')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the stage-2 checkpoint.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training.')
    parser.add_argument('--n_steps', type=int, default=20000,
                        help='Number of training steps.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(800, 608), help='Training resolution as width,height.')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run a short sanity check only.')
    parser.add_argument('--save_ckpt_every', type=int, default=2000,
                        help='Save checkpoint every N steps.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    return args


class Stage2Trainer:
    def __init__(self, megadepth_root_path, stage1_ckpt, ckpt_save_path,
                 batch_size=1, n_steps=20000, lr=3e-4, gamma_steplr=0.5,
                 training_res=(800, 608), device_num='0', dry_run=False,
                 save_ckpt_every=2000):
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = VUDNetModel(pretrained=False, use_desc_adapter=True).to(self.dev)
        self.net.load_state_dict(torch.load(stage1_ckpt, map_location=self.dev), strict=False)

        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=15000, gamma=gamma_steplr)
        
        TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
        TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"
        TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
        npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
        

        data = torch.utils.data.ConcatDataset([
            MegaDepthDataset(root_dir=TRAINVAL_DATA_SOURCE, npz_path=path)
            for path in tqdm(npz_paths, desc='[MegaDepth] Loading metadata')
        ])
        self.data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.data_iter = iter(self.data_loader)

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)
        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/stage2_' + time.strftime('%Y_%m_%d-%H_%M_%S'))

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

    def _freeze_stage1(self):
        for name, p in self.net.named_parameters():
            if name.startswith('stage') or name.startswith('proj') or name.startswith('adaptive_fusion') or name.startswith('block_fusion') or name.startswith('heatmap_head') or name.startswith('keypoint_head'):
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

    def train(self):
        self._freeze_stage1()
        self.net.train()
        d = None
        if self.data_iter is not None:
            d = next(self.data_iter)

        with tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if self.data_iter is not None:
                    try:
                        d = next(self.data_iter)
                    except StopIteration:
                        self.data_iter = iter(self.data_loader)
                        d = next(self.data_iter)

                if self.dry_run and i >= 2:
                    break

                loss_total = 0.0
                loss_count = 0
                for b in range(self.batch_size):
                    sample_data = self._get_sample_data(d, b)
                    sample_images = sample_data['images']
                    H_orig, W_orig = sample_images[0].shape[1:]
                    V = len(sample_images)

                    feats, hmaps, vars = [], [], []
                    for v in range(V):
                        img = sample_images[v]
                        if img.dim() == 3:
                            img = img.unsqueeze(0)
                        feat_v, _, hmap_v, var_v = self.net(img.to(self.dev))
                        feats.append(feat_v)
                        hmaps.append(hmap_v)
                        vars.append(var_v)

                    batch_points_dict, id_to_idx = generate_exclusive_subsets(sample_data)
                    subset_views_list = [5, 4, 3, 2]
                    subset_loss = []
                    for k in subset_views_list:
                        if k not in batch_points_dict:
                            continue
                        subset_ids, (corrs_k, vis_k) = batch_points_dict[k]
                        if corrs_k is None or corrs_k.shape[0] < 20:
                            continue
                        max_points = 4000
                        if corrs_k.shape[0] > max_points:
                            idx = torch.randperm(corrs_k.shape[0])[:max_points]
                            corrs_k = corrs_k[idx]
                            vis_k = vis_k[idx]
                        corrs_k = corrs_k.to(self.dev)
                        vis_k = vis_k.to(self.dev)
                        if k > len(subset_ids):
                            continue
                        f_inv_per_point, sigma_per_point = [], []
                        for v_local in range(k):
                            view_id = subset_ids[v_local]
                            view_idx = id_to_idx[view_id]
                            coords = corrs_k[:, v_local, :].to(self.dev)
                            f_inv_sample = sample_map_at_coords(feats[view_idx], coords, H_orig, W_orig)
                            sigma_sample = sample_map_at_coords(vars[view_idx], coords, H_orig, W_orig)
                            f_inv_per_point.append(f_inv_sample)
                            sigma_per_point.append(sigma_sample)
                        f_inv_k = torch.stack(f_inv_per_point, dim=1)
                        sigma_k = torch.stack(sigma_per_point, dim=1)
                        visibility = vis_k.bool() if vis_k is not None else torch.ones((corrs_k.shape[0], k), dtype=torch.bool, device=self.dev)
                        loss_var_k, var_target = sigma_consistency_loss(f_inv_k, sigma_k, visibility)
                        subset_loss.append(loss_var_k)

                    if len(subset_loss) == 0:
                        continue

                    # descriptor consistency loss across views using sampled cluster structure
                    descriptor_loss = self._descriptor_cluster_loss(sample_data, feats, H_orig, W_orig, id_to_idx)
                    loss = sum(subset_loss) / max(1, len(subset_loss)) + 0.2 * descriptor_loss
                    loss_total += loss.detach()
                    loss_count += 1

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                    self.opt.step()
                    self.opt.zero_grad()
                    self.scheduler.step()

                if loss_count == 0:
                    continue

                if (i + 1) % self.save_ckpt_every == 0 or (self.dry_run and i == 0):
                    ckpt_path = os.path.join(self.ckpt_save_path, f'stage2_{i + 1}.pth')
                    torch.save(self.net.state_dict(), ckpt_path)
                    print('saved', ckpt_path)

                self.writer.add_scalar('Loss/total', loss_total / max(loss_count, 1), i)
                pbar.set_description('step={:d} loss={:.4f}'.format(i + 1, loss_total / max(loss_count, 1)))
                pbar.update(1)

    def _descriptor_cluster_loss(self, sample_data, feats, H_orig, W_orig, id_to_idx):
        batch_points_dict, _ = generate_exclusive_subsets(sample_data)
        losses = []
        for k in [5, 4, 3, 2]:
            if k not in batch_points_dict:
                continue
            subset_ids, (corrs_k, vis_k) = batch_points_dict[k]
            if corrs_k is None or corrs_k.shape[0] < 20:
                continue
            max_points = 1200
            if corrs_k.shape[0] > max_points:
                idx = torch.randperm(corrs_k.shape[0])[:max_points]
                corrs_k = corrs_k[idx]
                vis_k = vis_k[idx]
            corrs_k = corrs_k.to(self.dev)
            vis_k = vis_k.to(self.dev)
            visible = vis_k.bool()
            if visible.sum() == 0:
                continue

            anchor_feats = []
            for v_local in range(k):
                view_id = subset_ids[v_local]
                view_idx = id_to_idx[view_id]
                coords = corrs_k[:, v_local, :].to(self.dev)
                feat_v = sample_map_at_coords(feats[view_idx], coords, H_orig, W_orig)
                anchor_feats.append(feat_v)
            anchor_feats = torch.stack(anchor_feats, dim=1)  # [N, k, C]

            sample_limit = min(anchor_feats.shape[0], 256)
            anchor_feats = anchor_feats[:sample_limit]
            visible = visible[:sample_limit]
            if anchor_feats.shape[0] < 2:
                continue

            losses.append(supervised_contrastive_v2(
                anchor_feats,
                torch.arange(anchor_feats.shape[0], device=self.dev),
                temp=0.07,
                hard_mining_ratio=0.3,
                visibility=visible,
                max_views_per_point=min(3, k),
            ))

        if len(losses) == 0:
            return torch.zeros([], device=self.dev, requires_grad=True)
        return torch.stack(losses).mean()


if __name__ == '__main__':
    args = parse_arguments()
    trainer = Stage2Trainer(
        megadepth_root_path=args.megadepth_root_path,
        stage1_ckpt=args.stage1_ckpt,
        ckpt_save_path=args.ckpt_save_path,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every,
    )
    trainer.train()
