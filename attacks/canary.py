import numpy as np
from utils.shadow_utils import get_test
from utils.loader import load_dataset
from attacks.util_canary import canary_args, learning_canary
import os
import pickle
from attacks.lira import attack as lira_attack

def attack(dataset, n_shadows, test_logits):
    lira_score = lira_attack(dataset, n_shadows, test_logits)
    args = canary_args(dataset, n_shadows)
    n_samples = len(test_logits)
    shadow_dir = f"{dataset}/"
    shadow_model_info_list = []
    for s in range(n_shadows - 1):
        model_path = os.path.join(shadow_dir, str(s), "shadow_model.pt")
        keep_path = os.path.join(shadow_dir, str(s), "shadow_keep.npy")
        shadow_model_info_list.append({"model_path": model_path, "keep_path": keep_path})

    target_model_path = f"{args.dataset}/{args.n_shadows - 1}/shadow_model.pt"

    # load data
    train_data = load_dataset(args)

    # run
    final_score = learning_canary(
        lira_score, args, shadow_model_info_list, target_model_path, train_data
    )

    return final_score

