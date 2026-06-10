import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
import os
from modules.dataset.megadepth.utils import read_megadepth_gray, read_megadepth_depth, fix_path_from_d2net
import numpy.random as rnd



"""
从一系列megadepth 匹配图里得到5个同一个view 下对应点信息的方法如下
选一个 anchor 图 I0：

在 I0 上采样 grid 点
    用 warp_kpts：
        I0 → I1
        I0 → I2
        I0 → I3
        I0 → I4
    对每个点做：
        valid = valid_01 & valid_02 & valid_03 & valid_04
        👉 只保留 在5张图都可见的点

"""



class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_paths,
                 mode='train',
                 min_overlap_score = 0.2, #0.3,
                 max_overlap_score = 0.7, #1,
                 load_depth = True,
                 img_resize = (800,608), #or None
                 df=32,
                 img_padding=False,  
                 depth_padding=True, # MegaDepth 深度图的分辨率不统一
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        
        """
        npz 是某一个scene 的索引文件 里面包含
        {
            'pair_infos': [ (img_i, img_j, overlap_score), ... ],
            'image_paths': [...],
            'depth_paths': [...],
            'intrinsics': [...],
            'poses': [...]
        }
        
        当overlap = 0.1 时表示几乎没有共同区域。overlap = 0.9 时表示视角接近
        需要根据overlap 进行过滤，因为太低overlap 是噪声样本，学不到。而太高的overlap 又太简单，学不到泛化。
        
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        filename = os.path.basename(npz_paths[0])
        self.scene_id = filename.split('_')[0]
        self.load_depth = load_depth
        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            min_overlap_score = 0
                
        first_scene = np.load(npz_paths[0], allow_pickle=True)
        self.scene_info = {
            'image_paths': first_scene['image_paths'],
            'depth_paths': first_scene['depth_paths'],
            'intrinsics': first_scene['intrinsics'],
            'poses': first_scene['poses']}
          
        self.pair_infos = []
        for npz_path in npz_paths:
            scene = np.load(npz_path, allow_pickle=True)
            pair_infos = scene['pair_infos']
            self.pair_infos.extend(pair_infos) 
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score and pair_info[1] < max_overlap_score] 
            
        self.pair_infos = [pair_info for pair_info in self.pair_infos if pair_info[1] > min_overlap_score and pair_info[1] < max_overlap_score]

        # Create graph
        self.graph = defaultdict(list)
        self.overlap_dict = {}
        for (i, j), overlap, _ in self.pair_infos:
            self.graph[i].append((j, overlap))
            self.graph[j].append((i, overlap))
            
            self.overlap_dict[(i,j)] = overlap
            self.overlap_dict[(j,i)] = overlap


        # parameters for image resizing, padding and depthmap padding
        if mode == 'train':
            assert img_resize is not None #and img_padding and depth_padding

        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.depth_max_size = 2000 if depth_padding else None  # the upperbound of depthmaps size in megadepth.
        self.min_overlap_score = min_overlap_score
        self.max_overlap_score = max_overlap_score
        
        
        # 兼容 D2 Net 的路径格式
        for idx in range(len(self.scene_info['image_paths'])):
            self.scene_info['image_paths'][idx] = fix_path_from_d2net(self.scene_info['image_paths'][idx])

        for idx in range(len(self.scene_info['depth_paths'])):
            self.scene_info['depth_paths'][idx] = fix_path_from_d2net(self.scene_info['depth_paths'][idx])
        
    
    # Sample 5 views from 
    def sample_five_views(self, anchor):

        bins = [
            (0.10, 0.25),
            (0.25, 0.40),
            (0.40, 0.55),
            (0.55, 0.70)
        ]

        selected_views = []
        selected_overlaps = []

        neighbors = self.graph[anchor]

        for low, high in bins:

            candidates = [
                (j, overlap)
                for j, overlap in neighbors
                if low <= overlap < high
            ]

            if len(candidates) == 0:
                return None

            chosen_idx = np.random.randint(len(candidates))

            view_id, overlap = candidates[chosen_idx]

            selected_views.append(view_id)
            selected_overlaps.append(overlap)

        return [anchor] + selected_views, selected_overlaps

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, idx, subset_views=None):
        """
        subset_views: int, optional
            如果为 None，返回 5-view。
            如果为 4/3/2，则从采样到的 5-view 中随机选择 subset_views 个。
        """
        (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx % len(self)]
    
        anchor = idx0
        # 1. 先采样 5 张 view
        result = self.sample_five_views(anchor)
        if result is None:
            return self.__getitem__(
            np.random.randint(len(self)),
            subset_views=subset_views
            )
        ids_5, overlaps_5 = result
    
        # 2. 如果需要子集，从5-view里随机选择
        if subset_views is not None and subset_views < len(ids_5):

            # anchor固定保留
            remain_ids = ids_5[1:]
            remain_overlaps = overlaps_5

            selected_idx = np.random.choice(
                len(remain_ids),
                subset_views - 1,
                replace=False
            )

            ids = [anchor]
            overlaps = []

            for idx_ in selected_idx:
                ids.append(remain_ids[idx_])
                overlaps.append(remain_overlaps[idx_])

        else:
            ids = ids_5
            overlaps = overlaps_5
        
        

        # 3. 读取 images / depth / Ks / poses / scales / masks
        images, depths, Ks, poses, scales, masks = [], [], [], [], [], []

        for i in ids:
            img_path = osp.join(self.root_dir, self.scene_info['image_paths'][i])
            image, mask, scale = read_megadepth_gray(img_path, self.img_resize, self.df, self.img_padding, None)
            images.append(image)
            scales.append(scale)
            # 保存mask：如果img_padding为True则mask非空，否则创建全1 mask
            if mask is not None:
                masks.append(mask)
            else:
                # 如果没有padding，mask默认为全1（所有像素都有效）
                masks.append(torch.ones(image.shape[1:], dtype=torch.bool))

            if self.load_depth:
                depth = read_megadepth_depth(
                    osp.join(self.root_dir, self.scene_info['depth_paths'][i]),
                    pad_to=self.depth_max_size)
                depths.append(depth)

            Ks.append(torch.tensor(self.scene_info['intrinsics'][i], dtype=torch.float).reshape(3, 3))
            poses.append(self.scene_info['poses'][i])

        # 4. 计算相对变换
        T0 = poses[0]
        T_0to = []
        for i in range(len(poses)):
            Ti = poses[i]
            T = torch.tensor(np.matmul(Ti, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]
            T_0to.append(T)

        data = {
            'images': images,
            'image_masks': masks,  # 新增：有效区域掩码
            'depths': depths,
            'Ks': Ks,
            'T_0to': T_0to,
            'T': poses,
            'scales': scales,
            'dataset_name': 'MegaDepth',
            'scene_id': self.scene_id,
            'view_ids': ids,
            'all_5view_ids': ids_5  # 可以保留完整5-view的信息
        }
        return data