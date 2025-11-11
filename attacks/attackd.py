import numpy as np
from utils.shadow_utils import get_all_distill_shadows


def attack(dataset, n_shadows, test_losses):
    alpha = 0.5
    temp = 1.0
    weight = 1.0
    model_name = f"mse_alpha{alpha:.1f}_temp{temp:.1f}_weight{weight:.1f}"
    all_shadow_out_losses = get_all_distill_shadows(
        dataset, 64, attack_type="loss", model_name=model_name, data_type="shadow", imitate_type="out"
    )
    n_samples = all_shadow_out_losses.shape[1]
    final_score = []

    for j in range(n_samples):
        out_loss = all_shadow_out_losses[:, j, 0]
        compare = np.sum(test_losses[j][0] <= out_loss)
        count = np.mean(compare)
        ratio = count / len(out_loss)
        final_score.append(ratio)

    final_score = np.array(final_score)
    return final_score
