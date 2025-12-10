from utils.loader import load_config, load_dataset, load_labels
import os
import numpy as np


class RmiaArgs:
    def __init__(self):
        # canary parameters
        self.proportiontocut = 0.2  # proportion to cut for trim_mean
        self.temperature = 2.0  # temperature for softmax


def rmia_args(dataset, n_shadows):
    args = RmiaArgs()
    args.dataset = dataset
    args.n_shadows = n_shadows
    load_config(args)
    return args


def get_all_train_probs(args, dataset, n_shadows, online=False):
    all_train_probs = []
    all_keep = []
    train_ds = load_dataset(args, data_type="shadow", augment=False)
    labels = load_labels(train_ds)
    n_samples = len(labels)
    shadow_dir = f"{dataset}/"
    for s in range(n_shadows):
        # [n_samples, n_queries, n_classes]
        if online:
            softmax_scores = np.load(os.path.join(shadow_dir, str(s), "shadow_softmax_on_shadow.npy"))
        else:
            softmax_scores = np.load(os.path.join(shadow_dir, str(s), "target_softmax_on_shadow.npy"))
        keep = np.load(os.path.join(shadow_dir, str(s), "shadow_keep.npy"))
        # [n_samples, n_queries]
        correct_probs = softmax_scores[np.arange(n_samples), :, labels]
        all_train_probs.append(correct_probs)
        all_keep.append(keep)

    all_train_probs = np.array(all_train_probs)  # [n_shadows, n_samples, n_queries]
    all_keep = np.array(all_keep)  # [n_shadows, n_samples]
    return all_train_probs, all_keep, n_samples


def get_all_z_probs(args, dataset, n_shadows, online=False):
    all_z_probs = []
    all_keep = []
    test_ds = load_dataset(args, data_type="target", augment=False)
    labels = load_labels(test_ds)
    n_samples = len(labels)
    shadow_dir = f"{dataset}/"
    for s in range(n_shadows):
        if online:
            z_softmax_scores = np.load(os.path.join(shadow_dir, str(s), "shadow_softmax_on_target.npy"))
        else:
            z_softmax_scores = np.load(os.path.join(shadow_dir, str(s), "target_softmax_on_target.npy"))
        z_probs = z_softmax_scores[np.arange(n_samples), :, labels]
        all_z_probs.append(z_probs)
        keep = np.load(os.path.join(shadow_dir, str(s), "target_keep.npy"))
        all_keep.append(keep)

    all_z_probs = np.array(all_z_probs)  # [n_shadows, n_samples, n_queries]
    all_keep = np.array(all_keep)  # [n_shadows, n_samples]
    if online:
        return all_z_probs
    else:
        all_final_z_probs = []
        for i in range(n_samples):
            z_probs_out = all_z_probs[~all_keep[:, i], i, :]
            all_final_z_probs.append(z_probs_out)
        all_final_z_probs = np.array(all_final_z_probs)  # [n_samples, n_shadow, n_queries]
        # change the shape to # [n_shadows, n_samples, n_queries]
        all_final_z_probs = np.transpose(all_final_z_probs, (1, 0, 2))
        return all_final_z_probs


def get_target_z_probs(args):
    test_ds = load_dataset(args, data_type="target", augment=False)
    labels = load_labels(test_ds)
    n_samples = len(labels)
    target_softmax_path = f"{args.dataset}/256/shadow_softmax_on_target.npy"
    target_z_softmax_scores = np.load(target_softmax_path)
    # [n_samples, n_queries, n_classes]
    target_z_probs = target_z_softmax_scores[np.arange(n_samples), :, labels]
    return target_z_probs
