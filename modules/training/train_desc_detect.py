import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules.vudnetmodel import VUDNetModel
from modules.dataset.augmentation import AugmentationPipe
from modules.dataset.megadepth.megadepth_stage1 import MegaDepthStage1Dataset
from modules.dataset.megadepth.megadepth_warper import *
from modules.training.descriptor_loss import DescriptorStageLoss
from modules.training.detector_loss import DetectorLoss, compute_correspondence
from modules.training.rdd.models.soft_detect import SoftDetect
from modules.training.utils import *


"""
For training the descriptor stage, run the following command:
python -m modules.training.train_desc_detect --megadepth_root_path datasets --ckpt_save_path checkpoints/stage1 --batch_size 1 --n_steps 10000

For training the detector stage, run the following command:
add --train_detector --weights <descriptor_checkpoint>

"""

def parse_arguments():
    parser = argparse.ArgumentParser(description="Stage-1 descriptor + detector training")
    parser.add_argument('--megadepth_root_path', type=str, required=True,
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default=None,
                        help='Optional synthetic dataset root directory for descriptor training.')
    parser.add_argument('--ckpt_save_path', type=str, required=True,
                        help='Path to save checkpoints.')
    parser.add_argument('--model_name', type=str, default='vudnet_stage1',
                        help='Name prefix for saved checkpoints.')
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
    parser.add_argument('--matchability_loss_weight', type=float, default=1.0,
                        help='Weight for matchability supervision loss.')
    parser.add_argument('--stability_eps', type=float, default=1e-6,
                        help='Small epsilon to stabilize stability normalization.')
    parser.add_argument('--train_detector', action='store_true',
                        help='Run detector training instead of descriptor stage.')
    parser.add_argument('--detector_top_k', type=int, default=4096,
                        help='Top k keypoints to extract in detector training.')
    parser.add_argument('--detector_scores_th', type=float, default=0.1,
                        help='SoftDetect score threshold for detector training.')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to a checkpoint to load before training.')
    return parser.parse_args()


def get_score_map(kpts, softmax_temp=1.0):
    scores = F.softmax(kpts * softmax_temp, dim=1)[:, :64]
    B, _, H, W = scores.shape
    heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
    heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
    return heatmap


class Trainer:
    def __init__(self, args):
        self.args = args
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_detector = args.train_detector

        self.net = VUDNetModel(pretrained=True, use_desc_adapter=False).to(self.dev)
        if args.weights is not None:
            self._load_weights(args.weights)

        self.batch_size = args.batch_size
        self.steps = args.n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=15000, gamma=args.gamma_steplr)

        self.augmentor = None
        if args.synthetic_root_path is not None and os.path.isdir(args.synthetic_root_path):
            self.augmentor = AugmentationPipe(
                img_dir=args.synthetic_root_path,
                device=self.dev,
                load_dataset=True,
                batch_size=max(1, int(self.batch_size * 0.4)),
                out_resolution=args.training_res,
                warp_resolution=args.training_res,
                sides_crop=0.1,
                max_num_imgs=3000,
                num_test_imgs=5,
                photometric=True,
                geometric=True,
                reload_step=4000,
            )

        self.descriptor_loss = DescriptorStageLoss(
            stability_weight=args.stability_weight,
            variance_weight=args.variance_loss_weight,
            matchability_weight=args.matchability_loss_weight,
            stability_eps=args.stability_eps,
        )
        self.detector_loss = DetectorLoss(temperature=0.1, scores_th=args.detector_scores_th) if self.train_detector else None
        self.softdetect = SoftDetect(radius=2, top_k=args.detector_top_k, scores_th=args.detector_scores_th) if self.train_detector else None

        train_base_path = f"{args.megadepth_root_path}/train_data/megadepth_indices"
        trainval_data_source = f"{args.megadepth_root_path}/MegaDepth_v1"
        train_npz_root = f"{train_base_path}/scene_info_0.1_0.7"
        npz_paths = glob.glob(train_npz_root + '/*.npz')[:]

        data = torch.utils.data.ConcatDataset([
            MegaDepthStage1Dataset(root_dir=trainval_data_source, npz_path=path)
            for path in tqdm(npz_paths, desc='[MegaDepth] Loading metadata')
        ])
        self.data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        self.data_iter = iter(self.data_loader)

        os.makedirs(args.ckpt_save_path, exist_ok=True)
        os.makedirs(args.ckpt_save_path + '/logdir', exist_ok=True)
        self.save_ckpt_every = args.save_ckpt_every
        self.ckpt_save_path = args.ckpt_save_path
        self.writer = SummaryWriter(args.ckpt_save_path + f'/logdir/{args.model_name}_' + time.strftime('%Y_%m_%d-%H_%M_%S'))
        self.model_name = args.model_name
        self.dry_run = args.dry_run

    def _load_weights(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Checkpoint not found: {path}')
        ckpt = torch.load(path, map_location=self.dev)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        self.net.load_state_dict(ckpt, strict=False)

    def _save_checkpoint(self, step):
        suffix = 'detector' if self.train_detector else 'descriptor'
        target_path = os.path.join(self.ckpt_save_path, f'{self.model_name}_{suffix}_{step}.pth')
        torch.save(self.net.state_dict(), target_path)

    def _get_batch(self):
        if self.data_iter is None:
            return None
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        return batch

    def _descriptor_step(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.dev)

        p1, p2 = batch['image0'], batch['image1']
        positives_md_coarse = spvs_coarse(batch, 8)
        if any(len(x) < 30 for x in positives_md_coarse):
            return None, None

        feats1, var1, match1, _ = self.net.forward_with_aux(p1)
        feats2, var2, match2, _ = self.net.forward_with_aux(p2)

        loss, metrics = self.descriptor_loss(
            feats1, feats2, var1, var2, match1, match2, positives_md_coarse, batch
        )
        return loss, metrics

    def _detector_step(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.dev)

        p1, p2 = batch['image0'], batch['image1']
        feats1, var1, match1, scores_map1 = self.net.forward_with_aux(p1)
        feats2, var2, match2, scores_map2 = self.net.forward_with_aux(p2)

        pred0 = {
            'descriptor_map': F.interpolate(feats1, size=scores_map1.shape[-2:], mode='bilinear', align_corners=True),
            'scores_map': scores_map1,
        }
        pred1 = {
            'descriptor_map': F.interpolate(feats2, size=scores_map2.shape[-2:], mode='bilinear', align_corners=True),
            'scores_map': scores_map2,
        }

        correspondences, pred0_with_rand, pred1_with_rand = compute_correspondence(
            self.net,
            pred0,
            pred1,
            batch,
            softdetect=self.softdetect,
            debug=False
        )
        loss = self.detector_loss(correspondences, pred0_with_rand, pred1_with_rand)
        metrics = {'loss': loss.item(), 'acc_coarse': 0.0, 'acc_kp': 0.0, 'nb_coarse': 0}
        return loss, metrics

    def train(self):
        with tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if self.dry_run and i > 10:
                    break
                batch = self._get_batch()
                if batch is None:
                    break

                if self.train_detector:
                    loss, metrics = self._detector_step(batch)
                else:
                    loss, metrics = self._descriptor_step(batch)

                if loss is None:
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if (i + 1) % self.save_ckpt_every == 0:
                    print('saving iter ', i + 1)
                    self._save_checkpoint(i + 1)

                pbar.set_description(
                    f"Step {i+1}/{self.steps} loss {loss.item():.4f}"
                )
                pbar.update(1)

                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse', metrics.get('acc_coarse', 0.0), i)
                self.writer.add_scalar('Accuracy/keypoint', metrics.get('acc_kp', 0.0), i)
                self.writer.add_scalar('Count/matches_coarse', metrics.get('nb_coarse', 0), i)

                if not self.train_detector:
                    self.writer.add_scalar('Loss/descriptors', metrics.get('loss_ds', 0.0), i)
                    self.writer.add_scalar('Loss/variance', metrics.get('loss_variance', 0.0), i)
                    self.writer.add_scalar('Loss/matchability', metrics.get('loss_matchability', 0.0), i)
                    self.writer.add_scalar('Loss/keypoint_pos', metrics.get('loss_keypoint', 0.0), i)

        print('Training finished.')


if __name__ == '__main__':
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    trainer = Trainer(args)
    trainer.train()
