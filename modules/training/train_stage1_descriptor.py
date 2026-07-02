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
from modules.dataset.megadepth.megadepth_stage1 import MegaDepthStage1Dataset
from modules.dataset.megadepth.megadepth_warper import *

from modules.dataset.megadepth.utils import *
from tqdm import tqdm


"""
python -m modules.training.train_stage1_descriptor \
  --megadepth_root_path datasets \
  --ckpt_save_path checkpoints/stage1 \
  --batch_size 1 \
  --n_steps 10000 \
  --device_num 0
"""


def _compute_relative_pose_diff(T_0to1, eps=1e-6):
    """Compute a scalar pose-change proxy from both translation and rotation.

    This function accepts either a single 4x4 pose tensor or a batch of 4x4 poses
    with shape [B, 4, 4]. For batched input, it returns one scalar per sample.
    """
    if T_0to1.dim() == 2 and T_0to1.shape == (4, 4):
        T_0to1 = T_0to1.unsqueeze(0)
    elif T_0to1.dim() != 3 or T_0to1.shape[1:] != (4, 4):
        raise ValueError(f'T_0to1 must have shape [4,4] or [B,4,4], got {tuple(T_0to1.shape)}')

    trans_norm = torch.linalg.norm(T_0to1[:, :3, 3], dim=1).float()
    R = T_0to1[:, :3, :3]
    trace = torch.einsum('bij->b', R)  # sum of diagonal elements
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    rot_angle = torch.acos(cos_theta)

    trans_term = trans_norm / (trans_norm + 1.0)
    rot_term = rot_angle / (torch.pi + eps)
    pose_diff = 0.5 * trans_term + 0.5 * rot_term
    return pose_diff


def parse_arguments():
    parser = argparse.ArgumentParser(description="Stage-1 descriptor/heatmap/reliability training")
    parser.add_argument('--megadepth_root_path', type=str, required=True,
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default=None,
                        help='Optional synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save the checkpoints.')
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
    parser.add_argument('--save_ckpt_every', type=int, default=10000,
                        help='Save checkpoint every N steps.')
    parser.add_argument('--stability_weight', type=float, default=1.0,
                        help='Weight for stability-aware descriptor supervision.')
    parser.add_argument('--variance_loss_weight', type=float, default=1.0,
                        help='Weight for variance regression loss.')
    parser.add_argument('--stability_eps', type=float, default=1e-6,
                        help='Small epsilon to stabilize stability normalization.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    return args


class Stage1Trainer:
    def __init__(self, megadepth_root_path, synthetic_root_path, ckpt_save_path,
                 batch_size=1, n_steps=20000, lr=3e-4, gamma_steplr=0.5,
                 training_res=(800, 608), device_num='0', dry_run=False,
                 save_ckpt_every=2000, stability_weight=1.0, variance_loss_weight=1.0,
                 stability_eps=1e-6):
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = VUDNetModel(pretrained=True, use_desc_adapter=False).to(self.dev)
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=15000, gamma=gamma_steplr)

        self.augmentor = None
        if synthetic_root_path is not None and os.path.isdir(synthetic_root_path):
            self.augmentor = AugmentationPipe(
                img_dir=synthetic_root_path,
                device=self.dev,
                load_dataset=True,
                batch_size=max(1, int(batch_size * 0.4)),
                out_resolution=training_res,
                warp_resolution=training_res,
                sides_crop=0.1,
                max_num_imgs=3000,
                num_test_imgs=5,
                photometric=True,
                geometric=True,
                reload_step=4000,
            )

        TRAIN_BASE_PATH = f"{megadepth_root_path}/train_data/megadepth_indices"
        TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/MegaDepth_v1"
        TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
        npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]

        data = torch.utils.data.ConcatDataset([
            MegaDepthStage1Dataset(root_dir=TRAINVAL_DATA_SOURCE, npz_path=path)
            for path in tqdm(npz_paths, desc='[MegaDepth] Loading metadata')
        ])
        self.data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        self.data_iter = iter(self.data_loader)

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)
        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.stability_weight = stability_weight
        self.variance_loss_weight = variance_loss_weight
        self.stability_eps = stability_eps
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/stage1_' + time.strftime('%Y_%m_%d-%H_%M_%S'))


    def train(self):
        self.net.train()

        difficulty = 0.10
        p1s, p2s, H1, H2 = None, None, None, None
        d = None

        if self.augmentor is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

        if self.data_iter is not None:
            d = next(self.data_iter)

        with tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if not self.dry_run:
                    if self.data_iter is not None:
                        try:
                            d = next(self.data_iter)
                        except StopIteration:
                            self.data_iter = iter(self.data_loader)
                            d = next(self.data_iter)

                    if self.augmentor is not None:
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)

                    p1, p2 = d['image0'], d['image1']
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)

                if self.augmentor is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _, positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)

                with torch.no_grad():
                    """
                    if d is not None:
                        p1 = p1.mean(1, keepdim=True)
                        p2 = p2.mean(1, keepdim=True)
                    if self.augmentor is not None:
                        p1s = p1s.mean(1, keepdim=True)
                        p2s = p2s.mean(1, keepdim=True)
                    
                    if self.augmentor is not None:
                        p1 = torch.cat([p1s, p1], dim=0)
                        p2 = torch.cat([p2s, p2], dim=0)
                        positives_c = positives_s_coarse + positives_md_coarse
                    else:
                        positives_c = positives_md_coarse
                    """
                    positives_c = positives_md_coarse

                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                feats1, kpts1, hmap1, var1 = self.net(p1)
                feats2, kpts2, hmap2, var2 = self.net(p2)

                loss_items = []
                loss_var = torch.zeros((), device=self.dev)
                loss_ds = torch.zeros((), device=self.dev)
                loss_kp = torch.zeros((), device=self.dev)
                loss_kp_pos = torch.zeros((), device=self.dev)
                acc_coarse_0 = 0.0
                acc_coarse = 0.0
                nb_coarse = 0
                for b in range(len(positives_c)):
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]
                    h_feat, w_feat = feats1.shape[-2], feats1.shape[-1]
                    pts1 = torch.stack([
                        pts1[:, 0].clamp(0, w_feat - 1),
                        pts1[:, 1].clamp(0, h_feat - 1),
                    ], dim=-1)
                    pts2 = torch.stack([
                        pts2[:, 0].clamp(0, w_feat - 1),
                        pts2[:, 1].clamp(0, h_feat - 1),
                    ], dim=-1)

                    m1 = feats1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].permute(1, 0)
                    m2 = feats2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].permute(1, 0)

                    v1 = var1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()].squeeze(-1)
                    v2 = var2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()].squeeze(-1)

                    h1 = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]
                    h2 = hmap2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()]

                    if m1.shape[0] == 0 or m2.shape[0] == 0:
                        continue

                    # Stability target: higher for more stable points and lower for unstable ones.
                    # We use the descriptor agreement and the pose change as a simple proxy.
                    with torch.no_grad():
                        T_0to1 = d['T_0to1'].to(self.dev)
                        pose_diff = _compute_relative_pose_diff(T_0to1, eps=self.stability_eps).detach()
                        if pose_diff.ndim > 0 and pose_diff.numel() > 1:
                            pose_diff = pose_diff[b]
                        desc_diff = 1.0 - F.cosine_similarity(m1, m2, dim=-1).detach().clamp(-1.0, 1.0)
                        stability_target = 1.0 / (1.0 + desc_diff / pose_diff.clamp(min=self.stability_eps))
                        stability_target = stability_target.clamp(0.0, 1.0).detach()

                    # Regress the variance head to predict this stability score.
                    variance_target = stability_target.to(self.dev)
                    variance_pred = torch.clamp(v1 * 0.5 + v2 * 0.5, 0.0, 1.0)
                    loss_var = F.mse_loss(variance_pred, variance_target)

                    # The descriptor loss is strengthened for points with high stability.
                    stability_weight = variance_target
                    loss_ds = weighted_pairwise_descriptor_loss(
                        m1,
                        m2,
                        stability=stability_weight,
                        temp=0.07,
                        stability_weight=self.stability_weight,
                    )

                    cos_sim = (m1 * m2).sum(dim=1)
                    conf = torch.sigmoid(cos_sim / 0.1).detach()

                    loss_kp_pos1, _ = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, _ = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
                    loss_kp = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append((self.variance_loss_weight * loss_var).unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                acc_coarse = check_accuracy(m1, m2)
                nb_coarse = len(m1)
                if len(loss_items) > 0:
                    loss = torch.cat(loss_items, -1).mean()
                else:
                    loss = torch.zeros((), device=self.dev, requires_grad=True)
                loss_coarse = loss_ds.item()
                loss_var = loss_var.item()
                loss_l1 = loss_kp.item()
                loss_kp_pos = loss_kp_pos.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i + 1) % self.save_ckpt_every == 0:
                    print('saving iter ', i + 1)
                    torch.save(self.net.state_dict(), self.ckpt_save_path + f'/stage1_{i + 1}.pth')

                pbar.set_description(
                    'Loss: {:.4f} acc_c0 {:.3f} acc_c1 {:.3f} loss_c: {:.3f} loss_kp: {:.3f} #matches_c: {:d} loss_kp_pos: {:.3f}'.format(
                        loss.item(), acc_coarse_0, acc_coarse, loss_coarse, loss_l1, nb_coarse, loss_kp_pos)
                )
                pbar.update(1)

                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                self.writer.add_scalar('Loss/coarse', loss_coarse, i)
                self.writer.add_scalar('Loss/variance', loss_var, i)
                self.writer.add_scalar('Loss/reliability', loss_l1, i)
                self.writer.add_scalar('Loss/keypoint_pos', loss_kp_pos, i)
                self.writer.add_scalar('Count/matches_coarse', nb_coarse, i)


if __name__ == '__main__':
    args = parse_arguments()
    trainer = Stage1Trainer(
        megadepth_root_path=args.megadepth_root_path,
        synthetic_root_path=args.synthetic_root_path,
        ckpt_save_path=args.ckpt_save_path,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every,
        stability_weight=args.stability_weight,
        variance_loss_weight=args.variance_loss_weight,
        stability_eps=args.stability_eps,
    )
    trainer.train()
