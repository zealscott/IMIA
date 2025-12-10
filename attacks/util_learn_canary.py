import argparse
import os
import numpy as np
from utils.loader import load_config, load_model
import random
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.stats
import torch.nn.functional as F


class CanaryArgs:
    def __init__(self):
        # canary parameters
        self.iter = 5  # Steps of Adam per canary
        self.lr = 0.05
        self.eps = 2.0  # L∞ clamp radius, or None for no L∞ constraint
        self.ensemble_k = 24  # Num canaries to average per sample
        self.max_in_gpu = 64  # Num of 'IN' models to cache in GPU
        self.max_out_gpu = 64  # Num of 'OUT' models to cache in GPU
        self.opt = "adamw"
        self.weight_decay = 0.001


def canary_args(dataset, n_shadows):
    args = CanaryArgs()
    args.dataset = dataset
    args.n_shadows = n_shadows
    load_config(args)
    return args


def collect_all_logits(shadow_model_info_list, n_samples):
    all_conf_in = [[] for _ in range(n_samples)]
    all_conf_out = [[] for _ in range(n_samples)]
    in_model_indices = [[] for _ in range(n_samples)]
    out_model_indices = [[] for _ in range(n_samples)]
    for model_idx, info in enumerate(shadow_model_info_list):
        logits_path = os.path.join(os.path.dirname(info["model_path"]), "shadow_scaled_logits_on_shadow.npy")
        scaled_logits = np.load(logits_path)
        keep_arr = np.load(info["keep_path"]).astype(bool)
        if scaled_logits.ndim == 3:
            logit_vals = scaled_logits.mean(axis=(1, 2))
        elif scaled_logits.ndim == 2:
            logit_vals = scaled_logits.mean(axis=1)
        else:
            logit_vals = scaled_logits
        for i in range(n_samples):
            val = logit_vals[i]
            if keep_arr[i]:
                all_conf_in[i].append(val)
                in_model_indices[i].append(model_idx)
            else:
                all_conf_out[i].append(val)
                out_model_indices[i].append(model_idx)

    mu_in = [np.median(arr) for arr in all_conf_in]
    sigma_in = [np.std(arr) for arr in all_conf_in]
    mu_out = [np.median(arr) for arr in all_conf_out]
    sigma_out = [np.std(arr) for arr in all_conf_out]

    return (mu_in, sigma_in, mu_out, sigma_out), (in_model_indices, out_model_indices)


def load_subset_of_models(idx_list, max_to_load, args, shadow_model_info_list, device="cuda"):
    """
    Picks up to 'max_to_load' random models from idx_list,
    loads them into GPU memory, returns the list [model1, model2, ...].

    If idx_list is smaller than max_to_load, we just load them all.
    """
    if len(idx_list) == 0:
        return []
    chosen = random.sample(idx_list, min(len(idx_list), max_to_load))
    loaded = []
    for idx in chosen:
        info = shadow_model_info_list[idx]
        m = load_model(args)
        m.load_state_dict(torch.load(info["model_path"], map_location=device))
        m.to(device)
        m.eval()
        loaded.append(m)
    return loaded


def load_all_shadow_models(args, shadow_model_info_list, device="cuda"):
    models = []
    max_models = 128
    for info in shadow_model_info_list[:max_models]:
        m = load_model(args)
        m.load_state_dict(torch.load(info["model_path"], map_location=device))
        m.eval()
        m.to(device)
        models.append(m)
    print(f"Loaded {len(models)} shadow models")
    return models


def canary_optimize(
    x_init,
    y_label,
    in_models_gpu,
    out_models_gpu,
    args,
    device="cuda",
    steps=30,
    lr=0.05,
    eps=2.0,
):
    x_can = x_init.clone().detach().to(device)
    x_can.requires_grad_(True)

    # Adam or AdamW, as in the original code
    if hasattr(args, "opt") and args.opt.lower() == "adamw":
        optimizer = optim.AdamW([x_can], lr=lr, weight_decay=getattr(args, "weight_decay", 0.0))
    else:
        optimizer = optim.Adam([x_can], lr=lr, weight_decay=getattr(args, "weight_decay", 0.0))

    for step in range(steps):
        optimizer.zero_grad()

        total_loss = torch.zeros(1, device=device, requires_grad=True)

        # accumulate "IN" => we add the logit => wants to push x_can "down"
        for m_in in in_models_gpu:
            with torch.enable_grad():
                logits_in = m_in(x_can)
                loss_in = F.cross_entropy(logits_in, y_label)
                total_loss = total_loss + loss_in

        # accumulate "OUT" => subtract => push x_can "up"
        for m_out in out_models_gpu:
            with torch.enable_grad():
                logits_out = m_out(x_can)
                loss_out = F.cross_entropy(logits_out, y_label)
                total_loss = total_loss - loss_out

        total_loss.backward()
        optimizer.step()
        # print(f"Step {step} loss: {total_loss.item()}")

        # clamp
        with torch.no_grad():
            if eps is not None:
                delta = torch.clamp(x_can - x_init, -eps, eps)
                x_can = torch.clamp(x_init + delta, 0.0, 1.0)
            else:
                x_can.clamp_(0.0, 1.0)
        x_can.requires_grad_()

    # free GPU memory for these cached models
    for m in in_models_gpu:
        del m
    for m in out_models_gpu:
        del m
    torch.cuda.empty_cache()

    return x_can.detach()


def cal_lira_score(target_logit, mu_in, sigma_in, mu_out, sigma_out):
    logp_in = scipy.stats.norm.logpdf(target_logit, mu_in, sigma_in + 1e-30)
    logp_out = scipy.stats.norm.logpdf(target_logit, mu_out, sigma_out + 1e-30)
    score = logp_in - logp_out
    return score.mean()


def cal_logit(model, x, y):
    with torch.no_grad():
        output = model(x).cpu().numpy()
    predictions = output[0]
    predictions = predictions - np.max(predictions)
    probs = np.exp(predictions)
    probs = probs / np.sum(probs)
    py = probs[y]
    p_wrong = 1 - py

    logit = np.log(py + 1e-45) - np.log(p_wrong + 1e-45)

    return logit
