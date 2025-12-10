import argparse
import os
import time
import numpy as np
import pytorch_lightning as pl
import torch
from utils.loader import load_labels, load_dataset, load_model
from utils.metric import get_acc
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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
parser.add_argument("--n_shadows", default=257, type=int)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--pkeep", default=0.5, type=float)
parser.add_argument("--data_dir", default="/path/to/your/datasets", type=str)
parser.add_argument("--savedir", default="./", type=str)

args = parser.parse_args()
print(args)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")


def acc_rule(args):
    if args.dataset == "cifar10":
        args.imitate_acc = 0.6
    elif args.dataset == "cifar100":
        args.imitate_acc = 0.4
    elif args.dataset == "mnist" or args.dataset == "fmnist":
        args.imitate_acc = 0.8
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")


def train_and_save(train_type="shadow"):
    print(f"Training {train_type} model {args.shadow_id} ...")
    seed = np.random.randint(0, 1000000000)
    seed ^= int(time.time())
    pl.seed_everything(seed)

    data_ds = load_dataset(args, data_type=train_type)
    size = len(data_ds)
    np.random.seed(2025)

    # First handle shadow models to ensure half IN/OUT distribution
    keep_shadows = np.random.uniform(0, 1, size=(args.n_shadows, size))
    order_shadows = keep_shadows.argsort(0)
    keep_shadows = order_shadows < int(args.pkeep * args.n_shadows)
    keep = np.array(keep_shadows[args.shadow_id], dtype=bool)

    keep = keep.nonzero()[0]
    keep_bool = np.full((len(data_ds)), False)
    keep_bool[keep] = True

    train_ds = torch.utils.data.Subset(data_ds, keep)
    test_ds = torch.utils.data.Subset(data_ds, ~keep)
    print(f"train with {len(train_ds)} data")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    m = load_model(args).to(device)
    optim = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    savedir = os.path.join(args.savedir, f"{args.shadow_id}")
    os.makedirs(savedir, exist_ok=True)
    # Train
    for i in range(args.epochs):
        m.train()
        loss_total = 0
        pbar = tqdm(train_dl)
        for itr, (x, y) in enumerate(pbar):
            if x.size(0) == 1:
                continue
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(m(x), y)
            loss_total += loss
            pbar.set_postfix_str(f"loss: {loss:.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

    test_acc = get_acc(m, test_dl, device)
    if test_acc < acc_rule(args):
        print(f"test accuracy is too low, {test_acc:.4f}, re-train")
        # train_and_save(train_type)
    print(f"test accuracy: {test_acc:.4f}")
    torch.save(m.state_dict(), os.path.join(savedir, f"{train_type}_model.pt"))
    np.save(os.path.join(savedir, f"{train_type}_keep.npy"), keep_bool)
    print(f"saved {train_type} model for shadow_id={args.shadow_id}")


@torch.no_grad()
def inference_all(savedir, infer_type="shadow", data_type="target"):
    print(f"inferring {infer_type} model {args.shadow_id} ...")
    data_ds = load_dataset(args, data_type=data_type)
    data_dl = DataLoader(data_ds, batch_size=512, shuffle=False, num_workers=4)
    m = load_model(args)
    m.load_state_dict(torch.load(os.path.join(savedir, f"{infer_type}_model.pt")))
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
    # Save all outputs
    np.save(os.path.join(savedir, f"{infer_type}_logits_on_{data_type}.npy"), logits_n)
    np.save(os.path.join(savedir, f"{infer_type}_softmax_on_{data_type}.npy"), softmax_n)
    np.save(os.path.join(savedir, f"{infer_type}_scaled_logits_on_{data_type}.npy"), scaled_logits)
    np.save(os.path.join(savedir, f"{infer_type}_losses_on_{data_type}.npy"), losses_n)
    print(f"Saved all inference outputs for {infer_type} model {args.shadow_id} on {data_type}")


if __name__ == "__main__":
    train_types = ["shadow", "target"]
    run_savedir = os.path.join(args.savedir, str(args.shadow_id))
    for train_type in train_types:
        if not os.path.exists(os.path.join(run_savedir, f"{train_type}_keep.npy")):
            train_and_save(train_type)
        for data_type in ["shadow", "target"]:
            if not os.path.exists(os.path.join(run_savedir, f"{train_type}_logits_on_{data_type}.npy")):
                inference_all(run_savedir, train_type, data_type)
