import argparse
import os
import time
import numpy as np
import pytorch_lightning as pl
import torch
from util_imitation import imitate_acc_rule, imitate_out_train, select_pivot_data, imitate_in_train
from utils.loader import load_labels, load_dataset, load_model, load_config
from utils.metric import get_acc
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn as nn
import copy

parser = argparse.ArgumentParser()
# model parameters
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--model_type", default="resnet", type=str)
# mia parameters
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--n_queries", default=None, type=int)
parser.add_argument("--shadow_id", default=0, type=int)
parser.add_argument("--n_imitate_shadows", default=20, type=int)
parser.add_argument("--dataset", default="fmnist", type=str)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--data_dir", default="/path/to/your/datasets", type=str)
parser.add_argument("--savedir", default="./", type=str)
# imitation parameters
parser.add_argument("--temperature", default=1.0, type=float)
parser.add_argument("--alpha", default=1.0, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int)
parser.add_argument("--mse_distillation", default=False, type=bool)
parser.add_argument("--margin_weight", default=1.0, type=float)
parser.add_argument("--imitate_acc", default=0.5, type=float)

args = parser.parse_args()
load_config(args)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")


def train_and_save_imitate(model_type):
    print(f"Training {model_type} model {args.shadow_id} ...")
    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    pl.seed_everything(seed)

    #########################################################
    # prepare configuration
    imitate_acc_rule(args)

    #########################################################
    # prepare data
    #########################################################
    # Load target dataset for imitation
    # all target data are OUT in the non-adaptive setting
    data_ds = load_dataset(args, data_type="target")
    size = len(data_ds)

    # Load teacher model (target-attacked model)
    teacher_model = load_model(args).to(device)
    teacher_model.load_state_dict(torch.load(f"{args.dataset}/256/shadow_model.pt"))
    teacher_model.eval()

    # 1. Select fixed pivot data
    pivot_indices = select_pivot_data(data_ds, teacher_model, args, strategy="low_loss", device=device)
    pivot_bool = np.full(len(data_ds), False)
    pivot_bool[pivot_indices] = True  # Only pivot data

    # 2. Use half IN/OUT logic for nonpivot selection
    np.random.seed(2025)
    keep_shadows = np.random.uniform(0, 1, size=(args.n_imitate_shadows, size))
    order_shadows = keep_shadows.argsort(0)
    keep_shadows = order_shadows < int(args.pkeep * args.n_imitate_shadows)
    keep = np.array(keep_shadows[args.shadow_id], dtype=bool)

    keep = keep.nonzero()[0]
    keep_bool = np.full((len(data_ds)), False)
    keep_bool[keep] = True

    # Create datasets
    train_ds = Subset(data_ds, keep)
    test_ds = Subset(data_ds, ~keep)

    print(f"train with {len(train_ds)} nonpivot data...")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    savedir = os.path.join(args.savedir, f"{args.shadow_id}")
    os.makedirs(savedir, exist_ok=True)

    #########################################################
    # train out model
    #########################################################
    # imitate shadow model using nonpivot data
    warmup_model, imitate_out_model = imitate_out_train(teacher_model, train_dl, test_dl, device, args)

    test_acc = get_acc(imitate_out_model, test_dl, device)
    if test_acc < args.imitate_acc:  # Use fixed threshold instead of args.imitate_acc
        print(f"imitate out model test accuracy is too low, {test_acc:.4f}, re-train")
        return -1  # exit the training
    print(f"imitate out model test accuracy: {test_acc:.4f}")

    #########################################################
    # continue training in model with pivot data
    #########################################################
    # ramdonly select nonpivot data to combine with pivot data to make the total size of pivot data equal to pkeep * size
    pivot_ds = Subset(data_ds, pivot_indices)
    pivot_dl = DataLoader(pivot_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    imitate_in_model = imitate_in_train(warmup_model, pivot_dl, test_dl, device, args)
    test_acc = get_acc(imitate_in_model, test_dl, device)
    if test_acc < args.imitate_acc:  # Use fixed threshold instead of args.imitate_acc
        print(f"imitate in model test accuracy is too low, {test_acc:.4f}, re-train")
        return -1  # exit the training
    print(f"imitate in model test accuracy: {test_acc:.4f}")

    np.save(os.path.join(savedir, f"nonadapt_pivot_{model_type}.npy"), pivot_bool)
    np.save(os.path.join(savedir, f"nonadapt_nonpivot_{model_type}.npy"), keep_bool)
    torch.save(imitate_out_model.state_dict(), os.path.join(savedir, f"nonadapt_out_{model_type}.pt"))
    torch.save(imitate_in_model.state_dict(), os.path.join(savedir, f"nonadapt_in_{model_type}.pt"))
    print(f"saved non-adaptive {model_type} model for shadow_id={args.shadow_id}")
    return 1  # success


@torch.no_grad()
def inference_all(savedir, data_type, model_name):
    print(f"inferring {model_name} model {args.shadow_id} ...")
    data_ds = load_dataset(args, data_type=data_type)
    data_dl = DataLoader(data_ds, batch_size=512, shuffle=False, num_workers=4)
    m = load_model(args)
    m.load_state_dict(torch.load(os.path.join(savedir, f"{model_name}.pt")))
    m.to(device)
    m.eval()
    logits_n = []
    softmax_n = []
    losses_n = []
    for i in range(args.n_queries):
        logits = []
        softmaxes = []
        losses = []
        for x, y in tqdm(data_dl):
            x = x.to(device)
            y = y.to(device)
            outputs = m(x)
            logits.append(outputs.cpu().numpy())
            softmaxes.append(torch.softmax(outputs, dim=1).cpu().numpy())
            loss = F.cross_entropy(outputs, y, reduction="none")
            losses.append(loss.detach().cpu().numpy())
        logits_n.append(np.concatenate(logits))
        softmax_n.append(np.concatenate(softmaxes))
        losses_n.append(np.concatenate(losses))
    logits_n = np.stack(logits_n, axis=1)  # [n_samples, n_queries, n_classes]
    softmax_n = np.stack(softmax_n, axis=1)
    losses_n = np.stack(losses_n, axis=1)  # [n_samples, n_queries]
    # Scaled logits (LIRA style)
    predictions = logits_n - np.max(logits_n, axis=-1, keepdims=True)
    predictions = np.array(np.exp(predictions), dtype=np.float64)
    predictions = predictions / np.sum(predictions, axis=-1, keepdims=True)
    labels = load_labels(data_ds)
    COUNT = predictions.shape[0]
    y_true = predictions[np.arange(COUNT), :, labels[:COUNT]]
    predictions[np.arange(COUNT), :, labels[:COUNT]] = 0
    y_wrong = np.sum(predictions, axis=-1)
    scaled_logits = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)
    if np.isnan(scaled_logits).any():
        print(f"scaled_logits is nan, exit")
        return -1
    # Save all outputs
    np.save(os.path.join(savedir, f"{model_name}_confs_on_{data_type}.npy"), scaled_logits)
    print(f"Saved all inference outputs for {model_name} model {args.shadow_id} on {data_type}")
    return 1


if __name__ == "__main__":
    print("Training non-adaptive imitation model ...")
    args.mse_distillation = True

    print(args)

    model_name = "imia"
    run_savedir = os.path.join(args.savedir, str(args.shadow_id))
    if not os.path.exists(os.path.join(run_savedir, f"nonadapt_in_{model_name}.pt")):
        train_and_save_imitate(model_name)

    for data_type in ["shadow", "target"]:
        if not os.path.exists(os.path.join(run_savedir, f"nonadapt_out_{model_name}_confs_on_{data_type}.npy")):
            inference_all(run_savedir, data_type, f"nonadapt_out_{model_name}")
        if not os.path.exists(os.path.join(run_savedir, f"nonadapt_in_{model_name}_confs_on_{data_type}.npy")):
            inference_all(run_savedir, data_type, f"nonadapt_in_{model_name}")
