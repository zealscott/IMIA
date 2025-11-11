import numpy as np
import scipy.stats
from sklearn.mixture import GaussianMixture


def one_gaussian_ll(target_margins, in_margin, out_margin):
    """
    one Gaussian log likelihood
    use mean and std of IN/OUT to calculate the log pdf of target
    """
    _in_mean = np.median(in_margin, 0)  # there are many queries
    _in_std = np.std(in_margin, 0)
    _out_mean = np.median(out_margin, 0)
    _out_std = np.std(out_margin, 0)

    logp_in = scipy.stats.norm.logpdf(target_margins, _in_mean, _in_std + 1e-30)
    logp_out = scipy.stats.norm.logpdf(target_margins, _out_mean, _out_std + 1e-30)

    score = logp_in - logp_out
    return score.mean()


def fit_two_gaussians(target_margin, in_margin, out_margin):
    """Fit a mixture of two Gaussians to the data and get the log likelihood ratio"""
    gmm = GaussianMixture(n_components=2, random_state=0)

    gmm.fit(in_margin.reshape(-1, 1))
    logp_in = gmm.score_samples(target_margin.reshape(-1, 1))

    gmm.fit(out_margin.reshape(-1, 1))
    logp_out = gmm.score_samples(target_margin.reshape(-1, 1))

    score = logp_in - logp_out
    return score.mean()


def two_gaussian_ll(target_margins, in_margins, out_margins):
    """
    two Gaussian log likelihood
    use mean and std of IN/OUT to calculate the log pdf of target
    """
    n_augment = in_margins.shape[1]
    ll_ratios = []
    for a in range(n_augment):
        in_margin = in_margins[:, a]
        out_margin = out_margins[:, a]
        ll_ratios.append(fit_two_gaussians(target_margins[a], in_margin, out_margin))
    return np.mean(ll_ratios)


def plot_logits(all_logits, test_logits, in_i, out_i, j, i):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))

    # Plot histograms with KDE
    sns.histplot(all_logits[in_i, j, 0], label="in", alpha=0.5, kde=True, stat="density")
    sns.histplot(all_logits[out_i, j, 0], label="out", alpha=0.5, kde=True, stat="density")

    # Add vertical line for test logit
    plt.axvline(test_logits[j, 0], color="red", linestyle="--", label="test logit")

    plt.title(f"Logit Distribution for i={i}, j={j}")
    plt.xlabel("Logit Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"logits_{i}_{j}.png")
    plt.close()
