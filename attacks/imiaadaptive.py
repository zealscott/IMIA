import numpy as np
import scipy.stats
from utils.shadow_utils import get_all_shadow_models, get_all_adapt_imitate_shadows, get_all_nonadpt_imitate_shadows
from utils.shadow_utils import get_test
from utils.metric import compute_metric
from sklearn.metrics import roc_curve


def attack(dataset, n_shadows, test_logits):
    n_shadow = 10
    alpha = 0.5
    temp = 2.0
    # weight = 1.0
    weight = 4.6 if dataset == "cifar100" else 2.3  # np.log(100) = 4.6, np.log(10) = 2.3
    # weight = 10.0 if dataset == "cifar100" else 3.2 # np.sqrt(100) = 10.0, np.sqrt(10) = 3.2
    model_name = f"mse_alpha{alpha:.1f}_temp{temp:.1f}_weight{weight:.1f}"
    print(model_name)

    all_logits, all_keep, n_samples = get_all_shadow_models(
        dataset, n_shadows, attack_type="lira", data_type="shadow", model_type="shadow"
    )
    all_distill_logits, all_keep_distill = get_all_adapt_imitate_shadows(
        dataset, 64, model_name, data_type="shadow", imitate_type="out"
    )

    final_score = []
    for j in range(n_samples):
        shadow_in = all_logits[all_keep[:, j], j, :][:n_shadow]
        distill_out = all_distill_logits[~all_keep_distill[:, j], j, :][:n_shadow]

        in_mean = np.mean(shadow_in, axis=0)
        out_mean = np.mean(distill_out, axis=0)

        if in_mean.mean() < out_mean.mean():
            lira_score = 0.0
        else:
            lira_score = (out_mean - test_logits[j]) ** 2 - (in_mean - test_logits[j]) ** 2
            lira_score = lira_score.mean()
        final_score.append(lira_score)

    final_score = np.array(final_score)
    return final_score
