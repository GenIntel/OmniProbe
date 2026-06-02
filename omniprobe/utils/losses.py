import torch
import torch.nn as nn
from loguru import logger


def sig_loss(depth_pr, depth_gt, sigma=0.85, eps=0.001, only_mean=False):
    """
    SigLoss
        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.
        adapated from DINOv2 code

    Args:
        depth_pr (FloatTensor): predicted depth
        depth_gt (FloatTensor): groundtruth depth
        eps (float): to avoid exploding gradient
    """
    # ignore invalid depth pixels
    valid = depth_gt > 0
    depth_pr = depth_pr[valid]
    depth_gt = depth_gt[valid]

    g = torch.log(depth_pr + eps) - torch.log(depth_gt + eps)

    loss = g.pow(2).mean() - sigma * g.mean().pow(2)
    loss = loss.sqrt()
    return loss


class DepthLoss(nn.Module):
    def __init__(self, weight_sig=10.0, weight_grad=0.5, max_depth=10):
        # Depth loss adapted from DINOv2.
        super().__init__()
        self.sig_w = weight_sig
        self.grad_w = weight_grad
        self.max_depth = max_depth

    def forward(self, pred, target):
        # 0 out max depth so it gets ignored
        target[target > self.max_depth] = 0

        loss_s = self.sig_w * sig_loss(pred, target)
        loss_g = self.grad_w * gradient_loss(pred, target)
        return loss_s + loss_g


def gradient_loss(depth_pr, depth_gt, eps=0.001):
    """GradientLoss.

    Adapted from https://www.cs.cornell.edu/projects/megadepth/ and DINOv2 repo

    Args:
        depth_pr (FloatTensor): predicted depth
        depth_gt (FloatTensor): groundtruth depth
        eps (float): to avoid exploding gradient
    """
    depth_pr_downscaled = [depth_pr] + [
        depth_pr[..., :: 2 * i, :: 2 * i] for i in range(1, 4)
    ]
    depth_gt_downscaled = [depth_gt] + [
        depth_gt[..., :: 2 * i, :: 2 * i] for i in range(1, 4)
    ]

    gradient_loss = 0
    for depth_pr, depth_gt in zip(depth_pr_downscaled, depth_gt_downscaled):

        # ignore invalid depth pixels
        valid = depth_gt > 0
        N = valid.sum(dim=(-2, -1)).clamp(min=1)

        depth_pr_log = torch.log(depth_pr + eps)
        depth_gt_log = torch.log(depth_gt + eps)
        log_d_diff = depth_pr_log - depth_gt_log

        log_d_diff = torch.mul(log_d_diff, valid)

        v_gradient = torch.abs(log_d_diff[..., 0:-2, :] - log_d_diff[..., 2:, :])
        v_valid = torch.mul(valid[..., 0:-2, :], valid[..., 2:, :])
        v_gradient = torch.mul(v_gradient, v_valid)

        h_gradient = torch.abs(log_d_diff[..., :, 0:-2] - log_d_diff[..., :, 2:])
        h_valid = torch.mul(valid[..., :, 0:-2], valid[..., :, 2:])
        h_gradient = torch.mul(h_gradient, h_valid)

        gradient_loss += (
            (h_gradient.sum(dim=(-2, -1)) + v_gradient.sum(dim=(-2, -1))) / N
        ).mean()

    return gradient_loss


def angular_loss(snorm_pr, snorm_gt, mask, uncertainty_aware=False, eps=1e-4):
    """
    Angular loss with uncertainty aware component based on Bae et al.
    """
    # ensure mask is float and batch x height x width
    assert mask.ndim == 4, f"mask should be (batch x height x width) not {mask.shape}"
    mask = mask.squeeze(1).float()

    # compute correct loss
    if uncertainty_aware:
        assert snorm_pr.shape[1] == 4
        loss_ang = torch.cosine_similarity(snorm_pr[:, :3], snorm_gt, dim=1)
        loss_ang = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()

        # apply elu and add 1.01 to have a min kappa of 0.01 (similar to paper)
        kappa = torch.nn.functional.elu(snorm_pr[:, 3]) + 1.01
        kappa_reg = (1 + (-kappa * torch.pi).exp()).log() - (kappa.pow(2) + 1).log()

        loss = kappa_reg + kappa * loss_ang
    else:
        assert snorm_pr.shape[1] == 3
        loss_ang = torch.cosine_similarity(snorm_pr, snorm_gt, dim=1)
        loss = loss_ang.clamp(min=-1 + eps, max=1 - eps).acos()

    # compute loss over valid position
    loss_mean = loss[mask.bool()].mean()
    if loss_mean.isnan():
        logger.error("NaN angular loss detected")
        raise FloatingPointError("NaN angular loss detected")
    return loss_mean
