import numpy as np
import scipy.stats
from utils.shadow_utils import get_all_shadow_models
from utils.shadow_utils import get_test
from utils.metric import compute_metric
from sklearn.metrics import roc_curve


def attack(dataset, n_shadows, test_losses):
    all_losses, _, n_samples = get_all_shadow_models(
        dataset, n_shadows, attack_type="attackr", data_type="shadow", model_type="target"
    )
    final_score = []

    for j in range(n_samples):
        out_loss = all_losses[:, j]
        compare = np.sum(test_losses[j] <= out_loss, axis=0)
        count = np.mean(compare)
        ratio = count / len(out_loss)
        final_score.append(ratio)

    final_score = np.array(final_score)
    return final_score

