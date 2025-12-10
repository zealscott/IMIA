import numpy as np
import scipy.stats
from utils.shadow_utils import get_all_shadow_models
from utils.loader import load_dataset
from attacks.util_pmia import pmia_args, likelihood_ratio
from attacks.util_search import find_similar_images_faiss, find_similar_images_clip


def attack(dataset, n_shadows, test_logits):
    n_models = 127

    all_shadow_logits, _, n_shadow = get_all_shadow_models(dataset, n_shadows, attack_type="lira", data_type="shadow", model_type="target")
    all_target_logits, target_keep, n_target = get_all_shadow_models(dataset, n_shadows, attack_type="lira", data_type="target", model_type="target")

    args = pmia_args(dataset, n_shadows)
    
    target_ds = load_dataset(args, data_type="target", augment=False)
    # Get labels from training dataset
    target_labels = np.array([label for _, label in target_ds])
    num_classes = len(np.unique(target_labels))
    class_in_logits = [[] for _ in range(num_classes)]
    class_in_logits_mean = [[] for _ in range(num_classes)]
    class_in_logits_std = [[] for _ in range(num_classes)]
    
    for j in range(n_target):
        cur_data_in = all_target_logits[target_keep[:, j], j, :]
        cur_label = target_labels[j]
        class_in_logits[cur_label].append(cur_data_in[:n_models])
    
    for i in range(num_classes):
        class_in_logits[i] = np.concatenate(class_in_logits[i], axis=0)
        class_in_logits_mean[i] = np.median(class_in_logits[i], axis=0)
        class_in_logits_std[i] = np.std(class_in_logits[i], axis=0)


    shadow_ds = load_dataset(args, data_type="shadow", augment=False)
    shadow_labels = np.array([label for _, label in shadow_ds])
    final_score = []
    # score shape: [n_shadow_models, n_samples, n_queries]
    for j in range(n_shadow):
        cur_label = shadow_labels[j]
        in_mean = class_in_logits_mean[cur_label]
        in_std = class_in_logits_std[cur_label]
        
        logits_out = all_shadow_logits[:, j, :]
        out_mean = np.median(logits_out, axis=0)
        out_std = np.std(logits_out, axis=0)
        lira_score = likelihood_ratio(test_logits[j], in_mean, in_std, out_mean, out_std)
        final_score.append(lira_score)

    final_score = np.array(final_score)
    return final_score
