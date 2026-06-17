import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from modules.vudnet import VUDNet
from modules.dataset.megadepth.megadepth_warper import generate_pairwise_corrs_independent
from modules.dataset.megadepth.utils import read_megadepth_gray, read_megadepth_depth, fix_path_from_d2net
from modules.interpolator import InterpolateSparse2d


def extract_descriptors_at_points(model, img_path, pts_orig):
    """
    Extract L2-normalized descriptors at arbitrary original-image coordinates.
    pts_orig: numpy array [N,2] in original image pixel coordinates (x,y)
    Returns: numpy array [N, C]
    """
    device = model.dev

    # read and prepare image + depth scale using repo helper
    img, _, scale = read_megadepth_gray(img_path, resize=None, df=32, padding=False, augment_fn=None)
    # img: torch (C, H, W)

    # preprocess through model to get feature map
    x = img[None].to(device)
    x, rh, rw = model.preprocess_tensor(x)
    B, C, Hc, Wc = x.shape

    with torch.no_grad():
        M1, _, _, _ = model.net(x)
        M1 = F.normalize(M1, dim=1)

    # convert pts (numpy) in original image coords -> preprocessed coords
    if pts_orig.shape[0] == 0:
        return np.zeros((0, M1.shape[1]))

    pts = torch.tensor(pts_orig, dtype=torch.float32, device=device)
    # model.preprocess_tensor returns rh = H/_H and rw = W/_W
    # so preprocessed coords = original_coords / [rw, rh]
    pts_pre = pts.clone()
    pts_pre[:, 0] = pts_pre[:, 0] / rw
    pts_pre[:, 1] = pts_pre[:, 1] / rh

    # sample from feature map using utility consistent with repo
    from modules.dataset.megadepth.utils import sample_map_at_coords

    desc = sample_map_at_coords(M1, pts_pre.cpu(), int(x.shape[-2]), int(x.shape[-1]))
    desc = F.normalize(desc, dim=1)
    return desc.cpu().numpy()


def compute_dataset_similarity(data_root, model_weights=None, max_pairs=None):
    model = VUDNet(weights=model_weights).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).eval()

    #scene_files = [f for f in os.listdir(data_root) if f.endswith('.npz')]
    scene_files = [
                #"0015_0.1_0.3.npz",
                #"0015_0.3_0.5.npz",
                "0022_0.1_0.3.npz",
                "0022_0.3_0.5.npz",
                "0022_0.5_0.7.npz",
            ]
    sims = []
    pair_count = 0

    for scene_file in scene_files:
        scene = np.load(os.path.join(data_root, scene_file), allow_pickle=True)
        pairs = scene['pair_infos']
        intrinsics = scene['intrinsics']
        poses = scene['poses']
        image_paths = scene['image_paths']
        

        for idx in range(len(scene['depth_paths'])):
            scene['depth_paths'][idx] = fix_path_from_d2net(scene['depth_paths'][idx])

        for p in pairs:
            idx0, idx1 = p[0]
            print("pair count = ", pair_count)
            img0_path = os.path.join(data_root, image_paths[idx0])
            img1_path = os.path.join(data_root, image_paths[idx1])
            
            

            img0, mask0, scale0 = read_megadepth_gray(img0_path, resize=None, df=32, padding=False)
            img1, mask1, scale1 = read_megadepth_gray(img1_path, resize=None, df=32, padding=False)

            depth0 = read_megadepth_depth(os.path.join(data_root, "Undistorted_SfM", scene['depth_paths'][idx0]))
            depth1 = read_megadepth_depth(os.path.join(data_root, "Undistorted_SfM", scene['depth_paths'][idx1]))

            K0 = torch.tensor(intrinsics[idx0], dtype=torch.float32)
            K1 = torch.tensor(intrinsics[idx1], dtype=torch.float32)

            T0 = np.array(poses[idx0])
            T1 = np.array(poses[idx1])
            T0 = torch.tensor(T0, dtype=torch.float32)
            T1 = torch.tensor(T1, dtype=torch.float32)
            T0to0 = torch.eye(4, dtype=torch.float32)
            T0to1 = T1 @ torch.linalg.inv(T0)

            data = {
                'images': [img0, img1],
                'depths': [depth0, depth1],
                'Ks': [K0, K1],
                'T_0to': [T0to0, T0to1],
                'scales': [scale0, scale1]
            }

            corrs, vis = generate_pairwise_corrs_independent(data, 0, 1, scale=8)

            if corrs.shape[0] == 0:
                continue

            pts0 = corrs[:, 0, :].cpu().numpy()
            pts1 = corrs[:, 1, :].cpu().numpy()

            desc0 = extract_descriptors_at_points(model, img0_path, pts0)
            desc1 = extract_descriptors_at_points(model, img1_path, pts1)

            if desc0.shape[0] == 0 or desc1.shape[0] == 0:
                continue

            sim_vals = (desc0 * desc1).sum(axis=1)
            sims.append(sim_vals)

            pair_count += 1
            if max_pairs is not None and pair_count >= max_pairs:
                break

        if max_pairs is not None and pair_count >= max_pairs:
            break

    if len(sims) == 0:
        print('No correspondences found.')
        return None

    all_sims = np.concatenate(sims)
    mean_sim = all_sims.mean()
    median_sim = np.median(all_sims)

    print(f'Total correspondences: {all_sims.shape[0]}')
    print(f'Mean cosine similarity: {mean_sim:.4f}')
    print(f'Median cosine similarity: {median_sim:.4f}')

    return {
        'mean': float(mean_sim),
        'median': float(median_sim),
        'values': all_sims
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets/megadepth_test_1500')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--max_pairs', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    compute_dataset_similarity(args.data_root, model_weights=args.weights, max_pairs=args.max_pairs)
