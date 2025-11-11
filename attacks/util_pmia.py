import numpy as np
from utils.loader import load_dataset, load_config
import scipy.stats


class PmiaArgs:
    def __init__(self):
        pass


def pmia_args(dataset, n_shadows):
    args = PmiaArgs()
    args.dataset = dataset
    args.n_shadows = n_shadows
    load_config(args)
    return args


def likelihood_ratio(target_margins, _in_mean, _in_std, _out_mean, _out_std):
    """
    likelihood ratio attack
    use mean and std of IN/OUT to calculate the log pdf of target
    """

    logp_in = scipy.stats.norm.logpdf(target_margins, _in_mean, _in_std + 1e-30)
    logp_out = scipy.stats.norm.logpdf(target_margins, _out_mean, _out_std + 1e-30)

    score = logp_in - logp_out
    return score.mean()
