import numpy as np
from utils.shadow_utils import get_all_shadow_models


def attack(dataset, n_shadows, test_losses):
    all_losses, _, n_samples = get_all_shadow_models(
        dataset, n_shadows, attack_type="calibrate", data_type="shadow", model_type="target"
    )
    final_score = []

    for j in range(n_samples):
        out_loss = all_losses[:, j]
        mean_loss = np.mean(out_loss, axis=0)
        calibrate_loss = np.mean(test_losses[j] - mean_loss)
        final_score.append(-calibrate_loss)

    final_score = np.array(final_score)
    return final_score
