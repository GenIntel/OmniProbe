import numpy as np
import torch
import torch.nn.functional as F


def compute_cost_volume(tokens: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine-similarity cost volumes between per-frame dense tokens and query embeddings.

    Args:
        tokens: (B, T, C, H, W) dense backbone features.
        query_tokens: (B, N, C) query descriptors sampled at the query frame.
    Returns:
        Tensor of shape (B, T, N, H, W)
    """
    b, t, c, h, w = tokens.shape
    _, n, _ = query_tokens.shape

    tokens_norm = F.normalize(tokens, dim=2)
    query_norm = F.normalize(query_tokens, dim=-1)

    cost_volumes = []
    for frame_idx in range(t):
        frame_tokens = tokens_norm[:, frame_idx].view(b, c, h * w)
        scores = torch.einsum("bnc,bcp->bnp", query_norm, frame_tokens)
        cost_volumes.append(scores.view(b, n, h, w))
    return torch.stack(cost_volumes, dim=1)


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
):
    """
    Reference TAP-Vid metric computation.
    Adapted from https://github.com/gorkaydemir/fomo_point_tracking.
    See THIRD_PARTY_LICENSES.md for details.
    """

    metrics = {}
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError(f"Unknown query mode {query_mode}")

    query_frame = np.round(query_points[..., 0]).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)

    frac_within_all = []
    jaccard_all = []
    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        count_correct = np.sum(is_correct & evaluation_points, axis=(1, 2))
        count_visible = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible
        metrics[f"pts_within_{thresh}"] = frac_correct
        frac_within_all.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))

        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)

        metrics[f"jaccard_{thresh}"] = jaccard
        jaccard_all.append(jaccard)

    metrics["average_jaccard"] = np.mean(np.stack(jaccard_all, axis=1), axis=1)
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(frac_within_all, axis=1), axis=1
    )
    return metrics


class TapVidEvaluator:
    """Helper to accumulate TAP-Vid metrics across the dataset."""

    def __init__(self, zero_shot: bool = True):
        self.zero_shot = zero_shot
        self.reset()

    def reset(self):
        self.delta_avg = []
        self.delta_1 = []
        self.delta_2 = []
        self.delta_4 = []
        self.delta_8 = []
        self.delta_16 = []
        self.occlusion = []
        self.jaccard = []
        self.counter = 0

    def update(self, out_metrics):
        self.counter += 1
        self.delta_avg.append(out_metrics["average_pts_within_thresh"][0] * 100)
        self.delta_1.append(out_metrics["pts_within_1"][0] * 100)
        self.delta_2.append(out_metrics["pts_within_2"][0] * 100)
        self.delta_4.append(out_metrics["pts_within_4"][0] * 100)
        self.delta_8.append(out_metrics["pts_within_8"][0] * 100)
        self.delta_16.append(out_metrics["pts_within_16"][0] * 100)
        self.occlusion.append(out_metrics["occlusion_accuracy"][0] * 100)
        self.jaccard.append(out_metrics["average_jaccard"][0] * 100)

    def report(self):
        def mean(values):
            return sum(values) / max(len(values), 1)

        print(f"Mean delta_avg: {mean(self.delta_avg):.2f}")
        print(f"Mean delta_1:   {mean(self.delta_1):.2f}")
        print(f"Mean delta_2:   {mean(self.delta_2):.2f}")
        print(f"Mean delta_4:   {mean(self.delta_4):.2f}")
        print(f"Mean delta_8:   {mean(self.delta_8):.2f}")
        print(f"Mean delta_16:  {mean(self.delta_16):.2f}")
        print(f"Mean occ acc:   {mean(self.occlusion):.2f}")
        if not self.zero_shot:
            print(f"Mean Jaccard:   {mean(self.jaccard):.2f}")