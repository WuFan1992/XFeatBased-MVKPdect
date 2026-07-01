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
    parser.add_argument('--save_ckpt_every', type=int, default=1000,
                        help='Save checkpoint every N steps.')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    return args


class Stage1Trainer:
    def __init__(self, megadepth_root_path, synthetic_root_path, ckpt_save_path,
                 batch_size=1, n_steps=20000, lr=3e-4, gamma_steplr=0.5,
                 training_res=(800, 608), device_num='0', dry_run=False,
                 save_ckpt_every=2000):
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = VUDNetModel(pretrained=False, use_desc_adapter=False).to(self.dev)
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
        
        print(tqdm)
        print(type(tqdm))
        
        
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

                feats1, kpts1, hmap1, _ = self.net(p1)
                feats2, kpts2, hmap2, _ = self.net(p2)

                loss_items = []
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

                    h1 = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]
                    h2 = hmap2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()]

                    feats_pair = torch.cat([m1, m2], dim=0)
                    labels_pair = torch.arange(m1.shape[0], device=self.dev).repeat(2)
                    feats_pair = F.normalize(feats_pair, dim=-1)
                    loss_ds = supervised_contrastive_v2(feats_pair, labels_pair, temp=0.07, hard_mining_ratio=0.3)

                    cos_sim = (m1 * m2).sum(dim=1)
                    conf = torch.sigmoid(cos_sim / 0.1).detach()

                    loss_kp_pos1, _ = alike_distill_loss(kpts1[b], p1[b])
                    loss_kp_pos2, _ = alike_distill_loss(kpts2[b], p2[b])
                    loss_kp_pos = (loss_kp_pos1 + loss_kp_pos2) * 2.0
                    loss_kp = keypoint_loss(h1, conf) + keypoint_loss(h2, conf)

                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    loss_items.append(loss_kp_pos.unsqueeze(0))

                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                acc_coarse = check_accuracy(m1, m2)
                nb_coarse = len(m1)
                loss = torch.cat(loss_items, -1).mean()
                loss_coarse = loss_ds.item()
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
    )
    trainer.train()
