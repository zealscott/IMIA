import numpy as np
from utils.shadow_utils import get_all_distill_shadows
import scipy.stats


def attack(dataset, n_shadows, test_logits):
    n_imitate_shadows = 10
    model_name = "imia"
    all_shadow_out_logits = get_all_distill_shadows(
        dataset, n_imitate_shadows, attack_type="logits", model_name=model_name, data_type="shadow", imitate_type="out"
    )
    n_samples = all_shadow_out_logits.shape[1]
    final_score = []

    for j in range(n_samples):
        out_logits = all_shadow_out_logits[:, j, :][:n_imitate_shadows]
        # # sample 10 from out_loss
        # sample_loss = out_loss[np.random.choice(len(out_loss), 10, replace=False)]
        # count the number that the test loss is smaller than the mean of the out_loss
        glira_score = offline_lira(test_logits[j], out_logits)
        final_score.append(glira_score)

    final_score = np.array(final_score)
    return final_score


def offline_lira(target_margins, out_margin):
    """
    offline lira attack
    use mean and std of OUT to calculate the log pdf of target
    """
    _out_mean = np.median(out_margin, 0)
    _out_std = np.std(out_margin, 0)

    logp_out = scipy.stats.norm.logpdf(target_margins, _out_mean, _out_std + 1e-30)
    score = -logp_out
    return score.mean()