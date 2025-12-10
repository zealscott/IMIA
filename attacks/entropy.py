import numpy as np
import scipy.stats
from utils.shadow_utils import get_all_shadow_models
from utils.loader import load_dataset, load_labels
from attacks.util_rmia import rmia_args


def attack(dataset, n_shadows, target_softmax_scores):
    args = rmia_args(dataset, n_shadows)
    target_ds = load_dataset(args, data_type="shadow", augment=False)
    labels = load_labels(target_ds)
    n_classes = 100 if dataset == "cifar100" else 10
    final_score = []
    n_targets = target_softmax_scores.shape[0]

    for j in range(n_targets):
        prob_y = target_softmax_scores[j, :, labels[j]]
        prob_else = []
        for i in range(n_classes):
            if i != labels[j]:
                prob_else.append(target_softmax_scores[j, :, i])
        prob_else = np.array(prob_else)
        prob_else = np.clip(prob_else, 1e-4, 1 - 1e-4)
        label_entropy = -(1 - prob_y) * np.log(prob_y)
        non_label_entropy = np.sum(-(prob_else) * np.log(1 - prob_else), axis=0)
        entropy = label_entropy + non_label_entropy
        # Replace inf values with NaN
        entropy[~np.isfinite(entropy)] = np.nan
        entropy = np.nan_to_num(entropy, nan=0.0)
        score = -np.nanmean(entropy)
        final_score.append(score)
    final_score = np.array(final_score)
    return final_score
