import cv2
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch import nn
import matplotlib.pyplot as plt

from modules.dataset.megadepth.megadepth_warper import warp_kpts

def image_to_score_coords(kpts_img, image_shape, score_shape):

    H_img, W_img = image_shape
    H_score, W_score = score_shape

    kpts_score = kpts_img.clone()


    # image -> score pixel
    kpts_score[:,0] *= W_score / W_img
    kpts_score[:,1] *= H_score / H_img


    # score pixel -> [-1,1]

    kpts_norm = torch.zeros_like(kpts_score)

    kpts_norm[:,0] = (
        kpts_score[:,0] / (W_score-1)
    ) * 2 - 1


    kpts_norm[:,1] = (
        kpts_score[:,1] / (H_score-1)
    ) * 2 - 1


    return kpts_norm

def score_to_image_coords(kpts, image_shape):
    """
    Convert normalized score-map coordinates [-1,1]
    to resized image pixel coordinates.

    Args:
        kpts:
            Nx2
            normalized coordinates (x,y)

        image_shape:
            H,W

    Returns:
        Nx2 image coordinates
    """

    H, W = image_shape

    kpts_img = torch.zeros_like(kpts)

    kpts_img[:, 0] = (kpts[:, 0] / 2 + 0.5) * (W - 1)
    kpts_img[:, 1] = (kpts[:, 1] / 2 + 0.5) * (H - 1)

    return kpts_img

def image_to_depth_coords(kpts_img, image_shape, depth_shape):

    H_img,W_img=image_shape
    H_depth,W_depth=depth_shape


    kpts_depth=torch.zeros_like(kpts_img)

    # x coordinate
    kpts_depth[:,0] = (
        kpts_img[:,0]
        * W_depth
        / W_img
    )


    # y coordinate
    kpts_depth[:,1] = (
        kpts_img[:,1]
        * H_depth
        / H_img
    )


    return kpts_depth

def depth_to_score_coords(kpts_depth, depth_shape, score_shape):

    H_depth, W_depth = depth_shape
    H_score, W_score = score_shape

    kpts_img = kpts_depth.clone()

    # depth -> resized image
    kpts_img[:,0] *= W_score / W_depth
    kpts_img[:,1] *= H_score / H_depth


    # image -> [-1,1]

    kpts_score = torch.zeros_like(kpts_img)

    kpts_score[:,0] = (
        kpts_img[:,0] / (W_score-1)
    )*2-1

    kpts_score[:,1] = (
        kpts_img[:,1] / (H_score-1)
    )*2-1


    return kpts_score

def plot_keypoints(image, kpts, radius=2, color=(255, 0, 0)):
    image = image.cpu().detach().numpy() if isinstance(image, torch.Tensor) else image
    kpts = kpts.cpu().detach().numpy() if isinstance(kpts, torch.Tensor) else kpts

    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    out = np.ascontiguousarray(deepcopy(image))
    kpts = np.round(kpts).astype(int)

    for kpt in kpts:
        y0, x0 = kpt
        cv2.drawMarker(out, (x0, y0), color, cv2.MARKER_CROSS, radius)
    return out


def mutual_argmax(value, mask=None, as_tuple=True):
    max0 = value.max(dim=1, keepdim=True)
    max1 = value.max(dim=0, keepdim=True)
    valid_max0 = value == max0[0]
    valid_max1 = value == max1[0]
    mutual = valid_max0 * valid_max1
    if mask is not None:
        mutual = mutual * mask
    return mutual.nonzero(as_tuple=as_tuple)


def mutual_argmin(value, mask=None):
    return mutual_argmax(-value, mask)


def detect_keypoints_from_scores_map(scores_map, top_k=4096, threshold=0.1, kernel_size=5):
    B, C, H, W = scores_map.shape
    assert C == 1, 'scores_map must have one channel.'

    pooled = F.max_pool2d(scores_map, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    peaks = (scores_map == pooled) & (scores_map > threshold)

    keypoints = []
    score_dispersity = []
    scores = []
    for idx in range(B):
        coords = peaks[idx, 0].nonzero(as_tuple=False)
        if coords.shape[0] == 0:
            keypoints.append(torch.zeros((0, 2), device=scores_map.device))
            score_dispersity.append(torch.zeros((0,), device=scores_map.device))
            scores.append(torch.zeros((0,), device=scores_map.device))
            continue

        yx = coords[:, [1, 0]].float()
        scores_kpts = scores_map[idx, 0, coords[:, 0], coords[:, 1]]
        local_mean = F.avg_pool2d(scores_map[idx:idx + 1], kernel_size=kernel_size, stride=1, padding=kernel_size // 2)[0, 0]
        local_mean_kpts = local_mean[coords[:, 0], coords[:, 1]]
        dispersity = (scores_kpts - local_mean_kpts).clamp(min=0.0)

        if coords.shape[0] > top_k:
            order = torch.argsort(scores_kpts, descending=True)[:top_k]
            yx = yx[order]
            scores_kpts = scores_kpts[order]
            dispersity = dispersity[order]

        normalized = yx.clone()
        normalized[:, 0] = normalized[:, 0] / (W - 1) * 2 - 1
        normalized[:, 1] = normalized[:, 1] / (H - 1) * 2 - 1

        keypoints.append(normalized)
        score_dispersity.append(dispersity)
        scores.append(scores_kpts)

    return keypoints, score_dispersity, scores


def _warp_keypoints_to_other_image(kpts_wh, depth_src, depth_dst, T_src2dst, K_src, K_dst):
    if kpts_wh.numel() == 0:
        return kpts_wh, kpts_wh.new_zeros((0, 2)), torch.empty((0,), dtype=torch.long, device=kpts_wh.device), torch.empty((0,), dtype=torch.long, device=kpts_wh.device)

    orig = kpts_wh
    orig = orig.clamp(min=0)
    orig_scaled = orig.unsqueeze(0)
    depth_src = depth_src.unsqueeze(0)
    depth_dst = depth_dst.unsqueeze(0)
    K_src = K_src.unsqueeze(0)
    K_dst = K_dst.unsqueeze(0)
    T_src2dst = T_src2dst.unsqueeze(0)

    valid_mask, warped = warp_kpts(orig_scaled, depth_src, depth_dst, T_src2dst, K_src, K_dst)
    valid_mask = valid_mask[0]
    warped = warped[0]
    ids = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if ids.numel() == 0:
        return kpts_wh.new_zeros((0, 2)), kpts_wh.new_zeros((0, 2)), ids, ids

    kpts_src = kpts_wh[ids]
    warped_dst = warped[ids].clamp(min=0)
    return kpts_src, warped_dst, ids, ids

class DetectorLoss(nn.Module):
    def __init__(self, temperature=0.1, scores_th=0.1, peaky_weight=1, reprojection_weight=1, scoremap_weight=0.001):
        super().__init__()
        self.temperature = temperature
        self.scores_th = scores_th
        self.peaky_weight = peaky_weight
        self.reprojection_weight = reprojection_weight
        self.scoremap_weight = scoremap_weight
        self.PeakyLoss = PeakyLoss(scores_th=scores_th)
        self.ReprojectionLocLoss = ReprojectionLocLoss(scores_th=scores_th)
        self.ScoreMapRepLoss = ScoreMapRepLoss(temperature=temperature)

    def forward(self, correspondences, pred0_with_rand, pred1_with_rand):
        loss_peaky0 = self.PeakyLoss(pred0_with_rand)
        loss_peaky1 = self.PeakyLoss(pred1_with_rand)
        loss_peaky = (loss_peaky0 + loss_peaky1) / 2.0

        loss_reprojection = self.ReprojectionLocLoss(pred0_with_rand, pred1_with_rand, correspondences)
        loss_score_map_rp = self.ScoreMapRepLoss(pred0_with_rand, pred1_with_rand, correspondences)
        
        loss_kp = (
            loss_peaky * self.peaky_weight +
            loss_reprojection * self.reprojection_weight +
            loss_score_map_rp * self.scoremap_weight
        )
        return loss_kp

class PeakyLoss(object):
    def __init__(self, scores_th: float = 0.1):
        super().__init__()
        self.scores_th = scores_th

    def __call__(self, pred):
        loss_mean = 0
        CNT = 0
        for idx in range(len(pred['scores'])):
            n_original = len(pred['score_dispersity'][idx])
            scores_kpts = pred['scores'][idx][:n_original]
            valid = scores_kpts > self.scores_th
            if valid.sum() == 0:
                continue
            loss_peaky = pred['score_dispersity'][idx][valid]
            loss_mean += loss_peaky.sum()
            CNT += len(loss_peaky)
        return loss_mean / CNT if CNT != 0 else pred['scores_map'].new_tensor(0)

class ReprojectionLocLoss(object):
    def __init__(self, norm: int = 1, scores_th: float = 0.1):
        super().__init__()
        self.norm = norm
        self.scores_th = scores_th

    def __call__(self, pred0, pred1, correspondences):
        loss_mean = 0
        CNT = 0
        
        for idx in range(len(correspondences)):
            corr = correspondences[idx]
            if corr['correspondence0'] is None:
                continue
            if self.norm == 2:
                dist = corr['dist']
            elif self.norm == 1:
                dist = corr['dist_l1']
            else:
                raise TypeError('No such norm in correspondence.')

            ids0_d = corr['ids0_d']
            ids1_d = corr['ids1_d']

            scores0 = corr['scores0'].detach()[ids0_d]
            scores1 = corr['scores1'].detach()[ids1_d]
            valid = (scores0 > self.scores_th) & (scores1 > self.scores_th)
            reprojection_errors = dist[ids0_d, ids1_d][valid]
            loss_mean += reprojection_errors.sum()
            CNT += len(reprojection_errors)
        return loss_mean / CNT if CNT != 0 else pred0['scores_map'].new_tensor(0)

class ScoreMapRepLoss(object):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.radius = 2

    def __call__(self, pred0, pred1, correspondences):
        loss_mean = 0
        CNT = 0
        for idx in range(len(correspondences)):
            corr = correspondences[idx]
            if corr['correspondence0'] is None:
                continue

            scores_map0 = pred0['scores_map'][idx]
            scores_map1 = pred1['scores_map'][idx]
            kpts01 = corr['kpts01']
            kpts10 = corr['kpts10']
            scores_kpts10 = F.grid_sample(scores_map0.unsqueeze(0), kpts10.view(1, 1, -1, 2), mode='bilinear', align_corners=True)[0, 0, 0, :]
            scores_kpts01 = F.grid_sample(scores_map1.unsqueeze(0), kpts01.view(1, 1, -1, 2), mode='bilinear', align_corners=True)[0, 0, 0, :]
            s0 = scores_kpts01 * corr['scores0']
            s1 = scores_kpts10 * corr['scores1']

            similarity_map_01 = corr['similarity_map_01_valid']
            similarity_map_10 = corr['similarity_map_10_valid']
            # These are already detached, no need to detach again
            pmf01 = similarity_map_01
            pmf10 = similarity_map_10
            
            # Free up unnecessary data from correspondence dict early
            corr.pop('dist', None)
            corr.pop('dist_l1', None)
            corr.pop('kpts01', None)
            corr.pop('kpts10', None)
            
            # image0 -> image1
            N01 = pmf01.shape[0]

            pmf01_kpts = F.grid_sample(
                pmf01.unsqueeze(1),          # [N01,1,H,W]
                kpts01.view(N01,1,1,2),
                mode='bilinear',
                align_corners=True
            ).view(-1)


            # image1 -> image0
            N10 = pmf10.shape[0]

            pmf10_kpts = F.grid_sample(
                pmf10.unsqueeze(1),          # [N10,1,H,W]
                kpts10.view(N10,1,1,2),
                mode='bilinear',
                align_corners=True
            ).view(-1)
            
            # Clean up large tensors
            del similarity_map_01, similarity_map_10, pmf01, pmf10, kpts01, kpts10
            
            repetability01 = pmf01_kpts
            repetability10 = pmf10_kpts

            fs0 = repetability01
            fs1 = repetability10

            if s0.sum() != 0:
                loss01 = (1 - fs0) * s0 * len(s0) / s0.sum()
                loss_mean += loss01.sum()
                CNT += len(loss01)
            if s1.sum() != 0:
                loss10 = (1 - fs1) * s1 * len(s1) / s1.sum()
                loss_mean += loss10.sum()
                CNT += len(loss10)
            
            del pmf01_kpts, pmf10_kpts, s0, s1, repetability01, repetability10, fs0, fs1
            
            # Periodic cache cleanup
            if (idx + 1) % 2 == 0:
                torch.cuda.empty_cache()

        return loss_mean / CNT if CNT != 0 else pred0['scores_map'].new_tensor(0)

def compute_keypoints_distance(kpts0, kpts1, p=2, debug=False):
    """
    Args:
        kpts0: torch.tensor [M,2]
        kpts1: torch.tensor [N,2]
        p: (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm

    Returns:
        dist, torch.tensor [N,M]
    """
    dist = kpts0[:, None, :] - kpts1[None, :, :]  # [M,N,2]
    dist = torch.norm(dist, p=p, dim=2)  # [M,N]
    return dist

def compute_correspondence(
    model,
    pred0,
    pred1,
    batch,
    softdetect=None,
    radius=2,
    rand=False,
    train_gt_th=5,
    debug=False,
    chunk_size=64,
    sim_map_downsample=4,
):
    b, c, h, w = pred0['scores_map'].shape
    wh = pred0['scores_map'][0].new_tensor([[w - 1, h - 1]])

    pred0_with_rand = {k: v for k, v in pred0.items()}
    pred1_with_rand = {k: v for k, v in pred1.items()}
    pred0_with_rand['scores'] = []
    pred1_with_rand['scores'] = []
    pred0_with_rand['descriptors'] = []
    pred1_with_rand['descriptors'] = []
    pred0_with_rand['num_det'] = []
    pred1_with_rand['num_det'] = []

    if softdetect is not None:
        kpts0_list, score_disp0, scores0 = softdetect.detect_keypoints(pred0['scores_map'])
        kpts1_list, score_disp1, scores1 = softdetect.detect_keypoints(pred1['scores_map'])
    else:
        kpts0_list, score_disp0, scores0 = detect_keypoints_from_scores_map(pred0['scores_map'])
        kpts1_list, score_disp1, scores1 = detect_keypoints_from_scores_map(pred1['scores_map'])
    


    pred0_with_rand['keypoints'] = kpts0_list
    pred0_with_rand['score_dispersity'] = score_disp0
    pred1_with_rand['keypoints'] = kpts1_list
    pred1_with_rand['score_dispersity'] = score_disp1

    correspondences = []
    for idx in range(b):
        kpts0 = pred0['keypoints'][idx] if 'keypoints' in pred0 else kpts0_list[idx]
        kpts1 = pred1['keypoints'][idx] if 'keypoints' in pred1 else kpts1_list[idx]


        if rand:
            rand0 = torch.rand(len(kpts0), 2, device=kpts0.device) * 2 - 1
            rand1 = torch.rand(len(kpts1), 2, device=kpts1.device) * 2 - 1
            kpts0 = torch.cat([kpts0, rand0])
            kpts1 = torch.cat([kpts1, rand1])
            pred0_with_rand['keypoints'][idx] = kpts0
            pred1_with_rand['keypoints'][idx] = kpts1

        scores_map0 = pred0['scores_map'][idx]
        scores_map1 = pred1['scores_map'][idx]
        scores_kpts0 = F.grid_sample(scores_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2), mode='bilinear', align_corners=True).squeeze()
        scores_kpts1 = F.grid_sample(scores_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2), mode='bilinear', align_corners=True).squeeze()

        
        img_h, img_w = batch['image0'].shape[-2:]

        kpts0_img = score_to_image_coords(
            kpts0,
            (img_h,img_w)
        )
        kpts1_img = score_to_image_coords(
            kpts1,
            (img_h,img_w)
        )

        # image coordinate -> depth coordinate
        depth0 = batch['depth0'][idx]
        depth1 = batch['depth1'][idx]


        kpts0_wh_ = image_to_depth_coords(
            kpts0_img,
            (img_h, img_w),
            depth0.shape[-2:]
        )


        kpts1_wh_ = image_to_depth_coords(
            kpts1_img,
            (img_h, img_w),
            depth1.shape[-2:]
        )

        local_mask = compute_keypoints_distance(kpts0_wh_.detach(), kpts0_wh_.detach()) < radius
        valid_cnt = torch.sum(local_mask, dim=1)

        for i in torch.where(valid_cnt > 1)[0]:
            kpt_indices = torch.where(local_mask[i])[0]
            scs_max_idx = scores_kpts0[kpt_indices].argmax()
            tmp_mask = torch.ones(len(kpt_indices), dtype=torch.bool, device=kpts0.device)
            tmp_mask[scs_max_idx] = False
            suppressed = kpt_indices[tmp_mask]
            valid_cnt[suppressed] = 0

        valid_mask = valid_cnt > 0
        kpts0_wh = kpts0_wh_[valid_mask]
        kpts0 = kpts0[valid_mask]
        scores_kpts0 = scores_kpts0[valid_mask]
        pred0_with_rand['keypoints'][idx] = kpts0
        
        
    
        valid_mask = valid_mask[:len(pred0_with_rand['score_dispersity'][idx])]
        pred0_with_rand['score_dispersity'][idx] = pred0_with_rand['score_dispersity'][idx][valid_mask]
        pred0_with_rand['num_det'].append(valid_mask.sum())

        local_mask = compute_keypoints_distance(kpts1_wh_.detach(), kpts1_wh_.detach()) < radius
        valid_cnt = torch.sum(local_mask, dim=1)
        for i in torch.where(valid_cnt > 1)[0]:
            kpt_indices = torch.where(local_mask[i])[0]
            scs_max_idx = scores_kpts1[kpt_indices].argmax()
            tmp_mask = torch.ones(len(kpt_indices), dtype=torch.bool, device=kpts1.device)
            tmp_mask[scs_max_idx] = False
            suppressed = kpt_indices[tmp_mask]
            valid_cnt[suppressed] = 0

        valid_mask = valid_cnt > 0
        kpts1_wh = kpts1_wh_[valid_mask]
        kpts1 = kpts1[valid_mask]
        scores_kpts1 = scores_kpts1[valid_mask]
        pred1_with_rand['keypoints'][idx] = kpts1
        
        valid_mask = valid_mask[:len(pred1_with_rand['score_dispersity'][idx])]
        pred1_with_rand['score_dispersity'][idx] = pred1_with_rand['score_dispersity'][idx][valid_mask]
        pred1_with_rand['num_det'].append(valid_mask.sum())
        
        pred0_with_rand['scores'].append(scores_kpts0)
        pred1_with_rand['scores'].append(scores_kpts1)

        descriptor_map0 = pred0['descriptor_map'][idx]
        descriptor_map1 = pred1['descriptor_map'][idx]
        descriptor_map0 = F.normalize(descriptor_map0, dim=0)
        descriptor_map1 = F.normalize(descriptor_map1, dim=0)

        desc0 = F.grid_sample(descriptor_map0.unsqueeze(0), kpts0.view(1, 1, -1, 2), mode='bilinear', align_corners=True)[0, :, 0, :].t()
        desc1 = F.grid_sample(descriptor_map1.unsqueeze(0), kpts1.view(1, 1, -1, 2), mode='bilinear', align_corners=True)[0, :, 0, :].t()
        desc0 = F.normalize(desc0, dim=-1)
        desc1 = F.normalize(desc1, dim=-1)

        pred0_with_rand['descriptors'].append(desc0)
        pred1_with_rand['descriptors'].append(desc1)

        depth0 = batch['depth0'][idx]
        depth1 = batch['depth1'][idx]
        K0 = batch['K0'][idx]
        K1 = batch['K1'][idx]
        T_0to1 = batch['T_0to1'][idx]
        T_1to0 = batch['T_1to0'][idx]


        kpts0_wh_valid, kpts01_wh, ids0, ids0_out = _warp_keypoints_to_other_image(
            kpts0_wh, depth0, depth1, T_0to1, K0, K1
        )
        kpts1_wh_valid, kpts10_wh, ids1, ids1_out = _warp_keypoints_to_other_image(
            kpts1_wh, depth1, depth0, T_1to0, K1, K0
        )
        
        

        if kpts0_wh_valid.numel() == 0 or kpts1_wh_valid.numel() == 0:

            correspondences.append({'correspondence0': None, 'correspondence1': None, 'dist': kpts0_wh.new_tensor(0)})
            continue

        dist01 = compute_keypoints_distance(kpts0_wh_valid, kpts10_wh)
        dist10 = compute_keypoints_distance(kpts1_wh_valid, kpts01_wh)
        dist_l2 = (dist01 + dist10.t()) / 2.0
        mutual_min_indices = mutual_argmin(dist_l2)
        dist_mutual_min = dist_l2[mutual_min_indices]
        valid = dist_mutual_min.detach() < train_gt_th
        ids0_d = mutual_min_indices[0][valid]
        ids1_d = mutual_min_indices[1][valid]
        correspondence0 = ids0[ids0_d]
        correspondence1 = ids1[ids1_d]

        dist01_l1 = compute_keypoints_distance(kpts0_wh_valid, kpts10_wh, p=1)
        dist10_l1 = compute_keypoints_distance(kpts1_wh_valid, kpts01_wh, p=1)
        dist_l1 = (dist01_l1 + dist10_l1.t()) / 2.0

        # Memory-efficient chunked computation for score-map repeatability supervision.
        # Use a downsampled descriptor map to keep the per-step memory bounded.
        desc0_valid = desc0[ids0]  # [N_valid0, D]
        desc1_valid = desc1[ids1]  # [N_valid1, D]

        if sim_map_downsample > 1:
            sim_map0 = F.avg_pool2d(descriptor_map0.unsqueeze(0), kernel_size=sim_map_downsample, stride=sim_map_downsample).squeeze(0)
            sim_map1 = F.avg_pool2d(descriptor_map1.unsqueeze(0), kernel_size=sim_map_downsample, stride=sim_map_downsample).squeeze(0)
        else:
            sim_map0 = descriptor_map0
            sim_map1 = descriptor_map1

        hs, ws = sim_map0.shape[-2:]

        # Flatten descriptor maps for efficient matrix multiplication.
        descriptor_map0_flat = sim_map0.reshape(sim_map0.shape[0], hs * ws).t()  # [hs*ws, D]
        descriptor_map1_flat = sim_map1.reshape(sim_map1.shape[0], hs * ws).t()  # [hs*ws, D]
        
        # Compute similarity_map_01_valid in chunks
        similarity_map_01_valid_chunks = []
        for chunk_start in range(0, desc0_valid.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, desc0_valid.shape[0])
            desc_chunk = desc0_valid[chunk_start:chunk_end]  # [chunk_size, D]
            sim_chunk = (desc_chunk @ descriptor_map1_flat.t()) * 20  # [chunk_size, hs*ws]
            sim_chunk = sim_chunk.softmax(dim=-1)
            sim_chunk = sim_chunk.reshape(chunk_end - chunk_start, hs, ws).clamp(1e-6, 1 - 1e-6)
            similarity_map_01_valid_chunks.append(sim_chunk)
        
        similarity_map_01_valid = torch.cat(similarity_map_01_valid_chunks, dim=0).detach()
        
        # Compute similarity_map_10_valid in chunks
        similarity_map_10_valid_chunks = []
        for chunk_start in range(0, desc1_valid.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, desc1_valid.shape[0])
            desc_chunk = desc1_valid[chunk_start:chunk_end]  # [chunk_size, D]
            sim_chunk = (desc_chunk @ descriptor_map0_flat.t()) * 20  # [chunk_size, hs*ws]
            sim_chunk = sim_chunk.softmax(dim=-1)
            sim_chunk = sim_chunk.reshape(chunk_end - chunk_start, hs, ws).clamp(1e-6, 1 - 1e-6)
            similarity_map_10_valid_chunks.append(sim_chunk)
        
        similarity_map_10_valid = torch.cat(similarity_map_10_valid_chunks, dim=0).detach()
        
        # Clean up to prevent memory buildup
        del descriptor_map0_flat, descriptor_map1_flat, desc0_valid, desc1_valid, sim_map0, sim_map1
        del similarity_map_01_valid_chunks, similarity_map_10_valid_chunks
        torch.cuda.empty_cache()

        img_shape0 = batch['image0'].shape[-2:]
        img_shape1 = batch['image1'].shape[-2:]


        kpts01 = image_to_score_coords(
            kpts01_wh.detach(),
            img_shape1,
            pred1['scores_map'].shape[-2:]
        )


        kpts10 = image_to_score_coords(
            kpts10_wh.detach(),
            img_shape0,
            pred0['scores_map'].shape[-2:]
        )

        correspondences.append({
            'correspondence0': correspondence0,
            'correspondence1': correspondence1,
            'scores0': scores_kpts0[ids0].detach(),
            'scores1': scores_kpts1[ids1].detach(),
            'kpts01': kpts01,
            'kpts10': kpts10,
            'ids0': ids0,
            'ids1': ids1,
            'ids0_out': ids0_out,
            'ids1_out': ids1_out,
            'ids0_d': ids0_d,
            'ids1_d': ids1_d,
            'dist_l1': dist_l1.detach(),
            'dist': dist_l2.detach(),
            'similarity_map_01_valid': similarity_map_01_valid,
            'similarity_map_10_valid': similarity_map_10_valid,
        })
        


    return correspondences, pred0_with_rand, pred1_with_rand

class EmptyTensorError(Exception):
    pass
