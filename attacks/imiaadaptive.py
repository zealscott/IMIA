import numpy as np
import scipy.stats
from utils.shadow_utils import get_all_shadow_models, get_all_adapt_imitate_shadows, get_all_nonadpt_imitate_shadows
from utils.shadow_utils import get_test
from utils.metric import compute_metric
from sklearn.metrics import roc_curve


def attack(dataset, n_shadows, test_logits):
    n_imitate_shadows = 10
    model_name = "imia"
    print(f"Using IMIA model: {model_name}")

    all_logits, all_keep, n_samples = get_all_shadow_models(
        dataset, n_shadows, attack_type="lira", data_type="shadow", model_type="shadow"
    )
    all_imitate_logits, all_keep_imitate = get_all_adapt_imitate_shadows(
        dataset, n_imitate_shadows, model_name, data_type="shadow", imitate_type="out"
    )

    final_score = []
    for j in range(n_samples):
        imitate_in = all_logits[all_keep[:, j], j, :][:n_imitate_shadows]
        imitate_out = all_imitate_logits[~all_keep_imitate[:, j], j, :][:n_imitate_shadows]

        in_mean = np.mean(imitate_in, axis=0)
        out_mean = np.mean(imitate_out, axis=0)

        if in_mean.mean() < out_mean.mean():
            lira_score = 0.0
        else:
            lira_score = (out_mean - test_logits[j]) ** 2 - (in_mean - test_logits[j]) ** 2
            lira_score = lira_score.mean()
        final_score.append(lira_score)

    final_score = np.array(final_score)
    return final_score
