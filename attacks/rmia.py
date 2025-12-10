import numpy as np
from attacks.util_rmia import get_all_z_probs, get_all_train_probs, rmia_args, get_target_z_probs
import os
from scipy.stats import trim_mean
from utils.loader import load_labels, load_dataset


def attack(dataset, n_shadows, target_softmax_scores):
    args = rmia_args(dataset, n_shadows)
    # compute the liklihood ratio of population z
    all_z_probs = get_all_z_probs(args, dataset, n_shadows, online=False)
    target_z_probs = get_target_z_probs(args)
    print(f"all_z_probs shape: {all_z_probs.shape}, target_z_probs shape: {target_z_probs.shape}")
    # Compute trimmed mean for z across all shadow models
    # all_z_logits shape: [n_shadows, n_samples, n_augment]
    z_probs_avg = trim_mean(all_z_probs, args.proportiontocut, axis=0)
    z_ratio_rev = 1 / (z_probs_avg / target_z_probs)  # [n_samples, n_augment]
    print(f"z_ratio_rev shape: {z_ratio_rev.shape}")
    # compute the liklihood ratio of population x
    all_x_probs, _, n_samples = get_all_train_probs(args, dataset, n_shadows, online=False)
    print(f"all_x_probs shape: {all_x_probs.shape}")
    # convert target_softmax_scores to probabilities
    train_ds = load_dataset(args, data_type="shadow", augment=False)
    labels = load_labels(train_ds)
    target_x_probs = target_softmax_scores[np.arange(n_samples), :, labels]

    final_score = []
    for i in range(n_samples):
        # Get logits for this sample
        # x_probs_in = all_x_probs[all_keep[:, i], i, :]
        # x_probs_out = all_x_probs[~all_keep[:, i], i, :]
        # x_probs = np.concatenate([x_probs_in, x_probs_out], axis=0)  # [n_in + n_out, n_augment]
        x_probs = all_x_probs[:, i, :]

        x_probs_avg = trim_mean(x_probs, args.proportiontocut, axis=0)  # [n_augment]
        # Compute ratio for this sample
        x_ratio = target_x_probs[i] / x_probs_avg  # [n_augment]
        # Compare x_ratio with z_ratio_rev for each sample
        ratio_counts = []
        for aug_idx in range(x_ratio.shape[0]):
            # Count how many z_ratio_rev are smaller than this x_ratio
            count = np.mean(x_ratio[aug_idx] > z_ratio_rev[:, aug_idx])
            ratio_counts.append(count / z_ratio_rev.shape[0])

        # Average across augmentations
        final_score.append(np.mean(ratio_counts))

    final_score = np.array(final_score)
    return final_score
