import numpy as np
from utils.shadow_utils import get_all_distill_shadows


def attack(dataset, n_shadows, test_losses):
    n_imitate_shadows = 10
    model_name = "imia"
    all_shadow_out_losses = get_all_distill_shadows(
        dataset, n_imitate_shadows, attack_type="loss", model_name=model_name, data_type="shadow", imitate_type="out"
    )
    n_samples = all_shadow_out_losses.shape[1]
    final_score = []

    for j in range(n_samples):
        out_loss = all_shadow_out_losses[:, j, :][:n_imitate_shadows]
        compare = np.sum(test_losses[j][0] <= out_loss)
        count = np.mean(compare)
        ratio = count / len(out_loss)
        final_score.append(ratio)

    final_score = np.array(final_score)
    return final_score
