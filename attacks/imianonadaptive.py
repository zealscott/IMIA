import numpy as np
import scipy.stats
from utils.shadow_utils import get_all_nonadpt_imitate_shadows, get_all_shadow_models
from utils.loader import load_dataset
from attacks.util_pmia import pmia_args, likelihood_ratio
from attacks.util_search import find_similar_images_faiss, find_similar_images_clip


def attack(dataset, n_shadows, test_logits):
    n_imitate_shadows = 10
    model_name = "imia"

    args = pmia_args(dataset, n_shadows)

    all_target_in_logits, pivot_masks, nonpivot_masks = get_all_nonadpt_imitate_shadows(
        dataset, n_imitate_shadows, model_name, data_type="target", imitate_type="in"
    )

    all_shadow_out_logits, _, _ = get_all_nonadpt_imitate_shadows(
        dataset, n_imitate_shadows, model_name, data_type="shadow", imitate_type="out"
    )

    print(f"all_shadow_out_logits shape: {all_shadow_out_logits.shape}")
    print(f"all_target_in_logits shape: {all_target_in_logits.shape}")
    print(f"pivot_masks shape: {pivot_masks.shape}")
    print(f"nonpivot_masks shape: {nonpivot_masks.shape}")

    target_ds = load_dataset(args, data_type="target", augment=False)
    # Get labels from training dataset
    target_labels = np.array([label for _, label in target_ds])
    num_classes = len(np.unique(target_labels))
    class_in_logits = [[] for _ in range(num_classes)]
    class_in_logits_mean = [[] for _ in range(num_classes)]

    # use pivot mask to get the in-distribution logits
    # pivot is IN the every shadow model
    for j in range(pivot_masks.shape[1]):
        cur_data_in = all_target_in_logits[pivot_masks[:, j], j, :][:n_imitate_shadows]
        cur_label = target_labels[j]
        class_in_logits[cur_label].append(cur_data_in)

    for i in range(num_classes):
        class_in_logits[i] = np.concatenate(class_in_logits[i], axis=0)
        class_in_logits_mean[i] = np.median(class_in_logits[i], axis=0)

    shadow_ds = load_dataset(args, data_type="shadow", augment=False)
    shadow_labels = np.array([label for _, label in shadow_ds])
    final_score = []
    # score shape: [n_shadow_models, n_samples, n_queries]
    for j in range(len(test_logits)):
        cur_label = shadow_labels[j]
        in_mean = class_in_logits_mean[cur_label]

        shadow_logits_out = all_shadow_out_logits[:, j, :][:n_imitate_shadows]
        out_mean = np.median(shadow_logits_out, axis=0)
        if in_mean.mean() < out_mean.mean():
            lira_score = 0.0
        else:
            lira_score = (out_mean - test_logits[j]) ** 2 - (in_mean - test_logits[j]) ** 2
            lira_score = lira_score.mean()
        final_score.append(lira_score)

    final_score = np.array(final_score)
    return final_score
