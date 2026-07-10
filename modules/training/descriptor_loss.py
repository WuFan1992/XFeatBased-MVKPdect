import torch
import torch.nn.functional as F
from torch import nn

from modules.training.losses import matchability_loss
from modules.training.utils import check_accuracy


def _compute_relative_pose_diff(T_0to1, eps=1e-6):
    if T_0to1.dim() == 2 and T_0to1.shape == (4, 4):
        T_0to1 = T_0to1.unsqueeze(0)
    elif T_0to1.dim() != 3 or T_0to1.shape[1:] != (4, 4):
        raise ValueError(f'T_0to1 must have shape [4,4] or [B,4,4], got {tuple(T_0to1.shape)}')

    trans_norm = torch.linalg.norm(T_0to1[:, :3, 3], dim=1).float()
    R = T_0to1[:, :3, :3]
    trace = torch.einsum('bij->b', R)
    cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0 + eps, 1.0 - eps)
    rot_angle = torch.acos(cos_theta)

    trans_term = trans_norm / (trans_norm + 1.0)
    rot_term = rot_angle / (torch.pi + eps)
    pose_diff = 0.5 * trans_term + 0.5 * rot_term
    return pose_diff




def _sample_hard_negative_descriptors(feature_map, positive_descs, positive_coords, h_feat, w_feat, num_samples, device):
    if feature_map is None or positive_descs is None:
        return None

    feature_map = feature_map.to(device)
    if positive_descs.dim() == 1:
        positive_descs = positive_descs.unsqueeze(0)
    if positive_coords is not None and positive_coords.dim() == 1:
        positive_coords = positive_coords.unsqueeze(0)

    positive_descs = positive_descs.to(device)
    positive_descs = F.normalize(positive_descs, dim=-1)

    feat_flat = feature_map.permute(1, 2, 0).reshape(-1, feature_map.size(0))
    feat_flat = F.normalize(feat_flat, dim=-1)

    num_points = positive_descs.size(0)
    if num_points == 0:
        return None

    total_positions = feat_flat.size(0)
    candidate_pool_size = min(max(256, 16 * max(1, num_samples)), total_positions)
    if total_positions <= candidate_pool_size:
        candidate_idx = torch.arange(total_positions, device=device)
    else:
        candidate_idx = torch.randperm(total_positions, device=device)[:candidate_pool_size]

    candidate_feats = feat_flat[candidate_idx]
    scores = candidate_feats @ positive_descs.t()
    scores = scores.t()

    if positive_coords is not None and positive_coords.numel() > 0:
        positive_coords = positive_coords.to(device).long()
        for i in range(num_points):
            pos_y = int(positive_coords[i, 1].item())
            pos_x = int(positive_coords[i, 0].item())
            if 0 <= pos_y < h_feat and 0 <= pos_x < w_feat:
                pos_idx = pos_y * w_feat + pos_x
                if 0 <= pos_idx < total_positions:
                    candidate_pos_mask = candidate_idx == pos_idx
                    if candidate_pos_mask.any():
                        scores[i, candidate_pos_mask] = -1e9

    if scores.size(1) == 0:
        return None

    k = min(max(1, num_samples), scores.size(1))
    _, topk_idx = torch.topk(scores, k=k, dim=1)
    neg_feats = candidate_feats[topk_idx]
    return neg_feats


def _build_heatmap_target(h_feat, w_feat, points, device, dtype=torch.float32):
    labels = torch.zeros((h_feat, w_feat), dtype=dtype, device=device)
    if points.numel() > 0:
        labels[points[:, 1].long(), points[:, 0].long()] = 1.0
    return labels


def stability_weighted_hard_negative_descriptor_loss(
    m1,
    m2,
    stability,
    neg_m1=None,
    neg_m2=None,
    temp=0.07,
    stability_weight=1.0,
):
    if m1.dim() != 2 or m2.dim() != 2:
        raise RuntimeError('m1 and m2 must be 2D tensors')

    m1 = F.normalize(m1, dim=-1)
    m2 = F.normalize(m2, dim=-1)

    if neg_m1 is None or neg_m2 is None or neg_m1.numel() == 0 or neg_m2.numel() == 0:
        return weighted_pairwise_descriptor_loss(m1, m2, stability, temp=temp, stability_weight=stability_weight)

    neg_m1 = F.normalize(neg_m1, dim=-1)
    neg_m2 = F.normalize(neg_m2, dim=-1)

    if neg_m1.dim() == 2:
        neg_m1 = neg_m1.unsqueeze(1)
    if neg_m2.dim() == 2:
        neg_m2 = neg_m2.unsqueeze(1)

    stability = stability.to(m1.device).float().clamp(0.0, 1.0).reshape(-1)
    if stability.numel() != m1.size(0):
        stability = torch.ones(m1.size(0), device=m1.device, dtype=m1.dtype)

    weights = 1.0 + stability_weight * stability

    pos_sim_12 = (m1 * m2).sum(dim=-1) / temp
    neg_sim_12 = torch.einsum('mc,mkc->mk', m1, neg_m2) / temp
    logits_12 = torch.cat([pos_sim_12.unsqueeze(1), neg_sim_12], dim=1)
    targets_12 = torch.zeros(logits_12.size(0), dtype=torch.long, device=logits_12.device)
    loss12 = F.cross_entropy(logits_12, targets_12)

    pos_sim_21 = (m2 * m1).sum(dim=-1) / temp
    neg_sim_21 = torch.einsum('mc,mkc->mk', m2, neg_m1) / temp
    logits_21 = torch.cat([pos_sim_21.unsqueeze(1), neg_sim_21], dim=1)
    targets_21 = torch.zeros(logits_21.size(0), dtype=torch.long, device=logits_21.device)
    loss21 = F.cross_entropy(logits_21, targets_21)

    return ((loss12 + loss21) * weights).mean()


def weighted_pairwise_descriptor_loss(m1, m2, stability, temp=0.07, stability_weight=1.0):
    m1 = F.normalize(m1, dim=-1)
    m2 = F.normalize(m2, dim=-1)

    logits = (m1 @ m2.t()) / temp
    labels = torch.arange(logits.size(0), device=logits.device)

    loss12 = F.cross_entropy(logits, labels, reduction='none')
    loss21 = F.cross_entropy(logits.t(), labels, reduction='none')

    stability = stability.to(logits.device).float().clamp(0.0, 1.0)
    weights = 1.0 + stability_weight * stability
    loss = ((loss12 + loss21) * weights).mean()
    return loss


class DescriptorStageLoss(nn.Module):
    def __init__(
        self,
        stability_weight=1.0,
        variance_weight=1.0,
        matchability_weight=1.0,
        stability_eps=1e-6,
    ):
        super().__init__()
        self.stability_weight = stability_weight
        self.variance_weight = variance_weight
        self.matchability_weight = matchability_weight
        self.stability_eps = stability_eps

    def forward(self, feats1, feats2, var1, var2, match1, match2, positives_c, batch):
        loss_items = []
        acc_coarse_items = []

        loss_ds_val = 0.0
        loss_var_val = 0.0
        loss_match_val = 0.0
        acc_coarse_val = 0.0
        nb_coarse = 0

        for b in range(len(positives_c)):
            positives = positives_c[b]
            if len(positives) == 0:
                continue

            pts1, pts2 = positives[:, :2], positives[:, 2:]
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
            if m1.shape[0] == 0 or m2.shape[0] == 0:
                continue

            v1 = var1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()].squeeze(-1)
            v2 = var2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()].squeeze(-1)
            m1_match = match1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()].squeeze(-1)
            m2_match = match2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()].squeeze(-1)

            with torch.no_grad():
                T_0to1 = batch['T_0to1'][b].to(feats1.device)
                pose_diff = _compute_relative_pose_diff(T_0to1, eps=self.stability_eps).squeeze(0)
                desc_diff = 1.0 - F.cosine_similarity(m1, m2, dim=-1).detach().clamp(-1.0, 1.0)
                stability_target = 1.0 / (1.0 + desc_diff / pose_diff.clamp(min=self.stability_eps))
                stability_target = stability_target.clamp(0.0, 1.0).detach()

            variance_target = stability_target.to(feats1.device)
            variance_pred = torch.clamp(v1 * 0.5 + v2 * 0.5, 0.0, 1.0)
            loss_var = F.mse_loss(variance_pred, variance_target)

            stability_weight = variance_target
            neg_count = max(4, min(8, pts1.shape[0]))
            neg_m1 = _sample_hard_negative_descriptors(
                feats1[b],
                m1,
                pts1,
                h_feat,
                w_feat,
                num_samples=neg_count,
                device=feats1.device,
            )
            neg_m2 = _sample_hard_negative_descriptors(
                feats2[b],
                m2,
                pts2,
                h_feat,
                w_feat,
                num_samples=neg_count,
                device=feats2.device,
            )

            loss_ds = stability_weighted_hard_negative_descriptor_loss(
                m1,
                m2,
                stability=stability_weight,
                neg_m1=neg_m1,
                neg_m2=neg_m2,
                temp=0.07,
                stability_weight=self.stability_weight,
            )

            labels1 = _build_heatmap_target(h_feat, w_feat, pts1, feats1.device)
            labels2 = _build_heatmap_target(h_feat, w_feat, pts2, feats1.device)
            pred1 = match1[b, 0]
            pred2 = match2[b, 0]

            loss_match1 = matchability_loss(pred1, labels1)
            loss_match2 = matchability_loss(pred2, labels2)
            loss_match = 0.5 * (loss_match1 + loss_match2)

            loss_items.append((loss_ds + self.variance_weight * loss_var + self.matchability_weight * loss_match).unsqueeze(0))
            acc_coarse_items.append(check_accuracy(m1, m2))

            loss_ds_val += loss_ds.item()
            loss_var_val += loss_var.item()
            loss_match_val += loss_match.item()
            acc_coarse_val += acc_coarse_items[-1]
            nb_coarse += len(m1)

        if len(loss_items) > 0:
            loss = torch.cat(loss_items, -1).mean()
            acc_coarse = sum(acc_coarse_items) / len(acc_coarse_items)
        else:
            loss = torch.zeros((), device=feats1.device, requires_grad=True)
            acc_coarse = 0.0

        metrics = {
            'loss_ds': loss_ds_val / max(len(loss_items), 1),
            'loss_variance': loss_var_val / max(len(loss_items), 1),
            'loss_matchability': loss_match_val / max(len(loss_items), 1),
            'acc_coarse': acc_coarse,
            'nb_coarse': nb_coarse,
        }
        return loss, metrics
