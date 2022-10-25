import numpy as np
import torch
from einops import rearrange
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

from codebase.utils import utils


def apply_kmeans(opt, outputs, labels_true):
    input_phase = utils.phase_to_cartesian_coordinates(
        opt, outputs["phase"], outputs["norm_magnitude"]
    )

    input_phase = utils.tensor_to_numpy(input_phase)
    input_phase = rearrange(input_phase, "b p c h w -> b h w (c p)")

    num_clusters = int(torch.max(labels_true).item()) + 1

    labels_pred = (
        np.zeros((opt.input.batch_size, opt.input.image_height, opt.input.image_width))
        + num_clusters
    )

    # Run k-means on each image separately.
    for img_idx in range(opt.input.batch_size):
        in_phase = input_phase[img_idx]
        num_clusters_img = int(torch.max(labels_true[img_idx]).item()) + 1

        # Remove areas in which objects overlap before k-means analysis.
        label_idx = np.where(labels_true[img_idx].cpu().numpy() != -1)
        in_phase = in_phase[label_idx]

        # Run k-means.
        k_means = KMeans(n_clusters=num_clusters_img, random_state=opt.seed).fit(
            in_phase
        )

        # Create result image: fill in k_means labels & assign overlapping areas to class zero.
        cluster_img = (
            np.zeros((opt.input.image_height, opt.input.image_width)) + num_clusters
        )
        cluster_img[label_idx] = k_means.labels_
        labels_pred[img_idx] = cluster_img

    outputs["labels_pred"] = labels_pred
    return outputs


def calc_ari_score(opt, labels_true, labels_pred, with_background):
    ari = 0
    for idx in range(opt.input.batch_size):
        if with_background:
            area_to_eval = np.where(
                labels_true[idx] > -1
            )  # Remove areas in which objects overlap.
        else:
            area_to_eval = np.where(
                labels_true[idx] > 0
            )  # Remove background & areas in which objects overlap.

        ari += adjusted_rand_score(
            labels_true[idx][area_to_eval], labels_pred[idx][area_to_eval]
        )
    return ari / opt.input.batch_size
