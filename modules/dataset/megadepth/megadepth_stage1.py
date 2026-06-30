import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset

from modules.dataset.megadepth.utils import read_megadepth_gray, read_megadepth_depth, fix_path_from_d2net


class MegaDepthStage1Dataset(Dataset):
    """XFeat-style MegaDepth dataset for pairwise stage-1 training."""

    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 min_overlap_score=0.3,
                 max_overlap_score=1.0,
                 load_depth=True,
                 img_resize=(800, 608),
                 df=32,
                 img_padding=False,
                 depth_padding=True,
                 **kwargs):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('.')[0]
        self.load_depth = load_depth

        if mode == 'test' and min_overlap_score != 0:
            min_overlap_score = 0

        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_infos = self.scene_info['pair_infos'].copy()
        del self.scene_info['pair_infos']
        self.pair_infos = [
            pair_info for pair_info in self.pair_infos
            if pair_info[1] > min_overlap_score and pair_info[1] < max_overlap_score
        ]

        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None

        for idx in range(len(self.scene_info['image_paths'])):
            self.scene_info['image_paths'][idx] = fix_path_from_d2net(self.scene_info['image_paths'][idx])

        for idx in range(len(self.scene_info['depth_paths'])):
            self.scene_info['depth_paths'][idx] = fix_path_from_d2net(self.scene_info['depth_paths'][idx])

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx):
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx % len(self)]

        img_name0 = osp.join(self.root_dir, self.scene_info['image_paths'][idx0])
        img_name1 = osp.join(self.root_dir, self.scene_info['image_paths'][idx1])

        image0, mask0, scale0 = read_megadepth_gray(
            img_name0, self.img_resize, self.df, self.img_padding, None
        )
        image1, mask1, scale1 = read_megadepth_gray(
            img_name1, self.img_resize, self.df, self.img_padding, None
        )

        if self.load_depth:
            depth0 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx0]),
                pad_to=self.depth_max_size,
            )
            depth1 = read_megadepth_depth(
                osp.join(self.root_dir, self.scene_info['depth_paths'][idx1]),
                pad_to=self.depth_max_size,
            )
        else:
            depth0 = depth1 = torch.tensor([])

        K_0 = torch.tensor(self.scene_info['intrinsics'][idx0].copy(), dtype=torch.float).reshape(3, 3)
        K_1 = torch.tensor(self.scene_info['intrinsics'][idx1].copy(), dtype=torch.float).reshape(3, 3)

        T0 = self.scene_info['poses'][idx0]
        T1 = self.scene_info['poses'][idx1]
        T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,
            'depth0': depth0,
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,
            'T_1to0': T_1to0,
            'K0': K_0,
            'K1': K_1,
            'scale0': scale0,
            'scale1': scale1,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'pair_id': idx,
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
        }

        if mask0 is not None and mask1 is not None:
            data['mask0'] = mask0
            data['mask1'] = mask1

        return data
