import os
import numpy as np
import pickle
from typing import Tuple, Dict, List, Union
import glob
import shutil


def mia_suffix(attack_type):
    if (
        "loss" in attack_type
        or "attackr" in attack_type
        or "calibrate" in attack_type
        or "attackd" in attack_type
        or "rapid" in attack_type
        or "seqmia" in attack_type
    ):
        return "losses"
    elif "entropy" in attack_type or "rmia" in attack_type:
        return "softmax"
    elif "logits" in attack_type:
        return "logits"
    else:
        return "scaled_logits"


def mia_output(attack_type, data_type, model_type):
    suffix = mia_suffix(attack_type)
    return f"{model_type}_{suffix}_on_{data_type}.npy"


def get_test(
    dataset: str, n_shadows: int, attack_type: str = "lira", data_type: str = "shadow", model_type: str = "shadow"
):
    """
    Get the test models which are the last models in the first round.

    Args:
        dataset (str): Name of the dataset (e.g., 'cifar10')
        n_shadows (int): Number of shadow models

    Returns:
        - test_logits (np.ndarray): Array of shape (n_samples, n_augmentations) containing logits
            from the test model
        - test_masks (np.ndarray): Array of shape (n_samples, ) containing membership mask
            for the test model
    """
    print(f"load {dataset}/{n_shadows-1} as test model ...")
    res_file = mia_output(attack_type, data_type, model_type)
    test_logits = np.load(f"{dataset}/{n_shadows-1}/{res_file}")
    test_masks = np.load(f"{dataset}/{n_shadows-1}/{model_type}_keep.npy")

    return test_logits, test_masks


def get_all_shadow_models(
    dataset: str, n_shadow: int, attack_type: str = "lira", data_type: str = "shadow", model_type: str = "shadow"
):
    """
    Load the keep mask and scores of the shadow models.

    Args:
        dataset (str): Name of the dataset (e.g., 'cifar10')
        n_shadow (int): Number of shadow models
        attack_type (str): Type of attack to determine which scores to load

    Returns:
        - shadow_logits (np.ndarray): Array of shape (n_shadow, n_samples, n_augmentations) containing
            logits from shadow models
        - shadow_masks (np.ndarray): Array of shape (n_shadow, n_samples) containing membership
            masks for shadow models
        - n_samples (int): Number of samples in the dataset
    """
    shadow_logits = []
    shadow_masks = []
    shadow_dir = f"{dataset}"

    for model_idx in range(n_shadow - 1):
        model_dir = os.path.join(shadow_dir, str(model_idx))
        res_file = mia_output(attack_type, data_type, model_type)
        try:
            keep_arr = np.load(os.path.join(model_dir, f"{model_type}_keep.npy"))
            logits_arr = np.load(os.path.join(model_dir, res_file))
            shadow_masks.append(keep_arr)
            shadow_logits.append(logits_arr)
        except:
            print(f"No keep.npy or {res_file} for model {model_idx}")

    shadow_logits = np.array(shadow_logits)
    shadow_masks = np.array(shadow_masks)

    n_samples = shadow_logits.shape[1]

    return shadow_logits, shadow_masks, n_samples


def get_all_nonadpt_imitate_shadows(dataset, n_imitate_shadows, model_name, data_type="shadow", imitate_type="out"):
    """
    Load the pivot and nonpivot mask and scores of the shadow models.
    """
    shadow_logits = []
    pivot_masks = []
    nonpivot_masks = []
    shadow_dir = f"{dataset}"
    model_type = f"{imitate_type}_{model_name}"

    for model_idx in range(n_imitate_shadows):
        model_dir = os.path.join(shadow_dir, str(model_idx))
        pivot_arr = np.load(os.path.join(model_dir, f"nonadapt_pivot_{model_name}.npy"))
        nonpivot_arr = np.load(os.path.join(model_dir, f"nonadapt_nonpivot_{model_name}.npy"))
        logits_arr = np.load(os.path.join(model_dir, f"nonadapt_{model_type}_confs_on_{data_type}.npy"))

        pivot_masks.append(pivot_arr)
        nonpivot_masks.append(nonpivot_arr)
        shadow_logits.append(logits_arr)

    pivot_masks = np.array(pivot_masks)
    nonpivot_masks = np.array(nonpivot_masks)
    shadow_logits = np.array(shadow_logits)

    return shadow_logits, pivot_masks, nonpivot_masks


def get_all_adapt_imitate_shadows(dataset, n_imitate_shadows, model_name, data_type="shadow", imitate_type="out"):
    """
    Load the pivot and nonpivot mask and scores of the shadow models.
    """
    shadow_logits = []
    nonpivot_masks = []
    shadow_dir = f"{dataset}"
    model_type = f"{imitate_type}_{model_name}"

    for model_idx in range(n_imitate_shadows):
        model_dir = os.path.join(shadow_dir, str(model_idx))
        nonpivot_arr = np.load(os.path.join(model_dir, f"adapt_nonpivot_{model_name}.npy"))
        logits_arr = np.load(os.path.join(model_dir, f"adapt_{model_type}_confs_on_{data_type}.npy"))

        nonpivot_masks.append(nonpivot_arr)
        shadow_logits.append(logits_arr)

    nonpivot_masks = np.array(nonpivot_masks)
    shadow_logits = np.array(shadow_logits)
    return shadow_logits, nonpivot_masks


def get_all_distill_shadows(
    dataset, n_distill_shadows, attack_type, model_name, data_type="shadow", imitate_type="out"
):
    """
    the distill OUT models, used for attackd and GLiRA
    """
    shadow_logits = []
    shadow_dir = f"{dataset}"
    model_type = f"{imitate_type}_{model_name}"
    signal_type = "losses" if attack_type == "loss" else "confs"

    for model_idx in range(n_distill_shadows):
        model_dir = os.path.join(shadow_dir, str(model_idx))
        logits_arr = np.load(os.path.join(model_dir, f"nonadapt_{model_type}_{signal_type}_on_{data_type}.npy"))

        shadow_logits.append(logits_arr)

    shadow_logits = np.array(shadow_logits)

    return shadow_logits


def get_all_online_distill_shadows(
    dataset, n_distill_shadows, attack_type, model_name, data_type="shadow", imitate_type="out"
):
    """
    the distill OUT models, used for attackd and GLiRA
    """
    shadow_logits = []
    keep_masks = []
    shadow_dir = f"{dataset}"
    model_type = f"{imitate_type}_{model_name}"
    signal_type = "losses" if attack_type == "loss" else "confs"

    for model_idx in range(n_distill_shadows):
        model_dir = os.path.join(shadow_dir, str(model_idx))
        logits_arr = np.load(os.path.join(model_dir, f"adapt_{model_type}_{signal_type}_on_{data_type}.npy"))
        keep_arr = np.load(os.path.join(model_dir, f"adapt_nonpivot_{model_name}.npy"))

        shadow_logits.append(logits_arr)
        keep_masks.append(keep_arr)

    shadow_logits = np.array(shadow_logits)
    keep_masks = np.array(keep_masks)

    return shadow_logits, keep_masks