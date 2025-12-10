import numpy as np
from utils.shadow_utils import get_all_shadow_models


def attack(dataset, n_shadows, test_losses):
    _, _, n_samples = get_all_shadow_models(dataset, n_shadows, attack_type="loss", data_type="shadow", model_type="target")
    final_score = []
    for j in range(n_samples):
        final_score.append(-np.mean(test_losses[j]))
    final_score = np.array(final_score)
    return final_score
