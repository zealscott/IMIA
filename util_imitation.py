import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from utils.loader import load_dataset, load_labels, load_model
from tqdm import tqdm
import numpy as np
import copy
from utils.metric import get_acc



def imitate_out_train(teacher_model, train_loader, test_loader, device, args):
    """Train model using imitation from teacher model"""
    alpha = 0.5
    
    model = load_model(args).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    # Initialize loss functions
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    kldiv_criterion = nn.KLDivLoss(reduction="batchmean")

    warmup_model = None
    best_test_acc = 0
    best_model = None

    for i in range(args.epochs):
        model.train()
        loss_total = 0
        pbar = tqdm(train_loader)
        for itr, (x, y) in enumerate(pbar):
            if x.size(0) == 1:
                continue
            x, y = x.to(device), y.to(device)

            # Get student predictions
            student_logits = model(x)

            # Calculate CE loss
            ce_loss = ce_criterion(student_logits, y)

            # Warmup period: only use CE loss
            if i < args.warmup_epochs:
                loss = ce_loss
            else:
                # Get teacher predictions
                with torch.no_grad():
                    teacher_logits = teacher_model(x)

                # Calculate imitation loss (either MSE or KL)
                if args.mse_distillation:
                    # MSE imitation with margin focus
                    batch_indices = torch.arange(teacher_logits.size(0), device=teacher_logits.device)

                    # Find max incorrect label indices
                    tmp = teacher_logits.clone()
                    tmp[batch_indices, y] = float("-inf")
                    max_incorrect_indices = tmp.max(dim=1)[1]

                    # Create weight matrix
                    weight_matrix = torch.ones_like(student_logits)
                    weight_matrix[batch_indices, y] = args.margin_weight
                    weight_matrix[batch_indices, max_incorrect_indices] = args.margin_weight

                    # Weighted MSE loss with normalization to match CE scale
                    # apply temperature to soft logits, helpful for CIFAR100/CIFAR10
                    student_logits = student_logits / args.temperature
                    teacher_logits = teacher_logits / args.temperature
                    imitate_loss = mse_criterion(student_logits * weight_matrix, teacher_logits * weight_matrix)

                    # Normalize MSE loss to be in similar scale as CE loss
                    avg_weight = weight_matrix.mean()
                    imitate_loss = imitate_loss / avg_weight
                else:
                    # KL divergence based imitation
                    imitate_loss = kldiv_criterion(
                        F.log_softmax(student_logits / args.temperature, dim=1),
                        F.softmax(teacher_logits / args.temperature, dim=1),
                    ) * (args.temperature**2)

                if np.isnan(imitate_loss.item()):
                    raise ValueError("imitate_loss is nan")

                # Combine losses using alpha
                loss = alpha * imitate_loss + (1.0 - alpha) * ce_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_total += loss.item()
            if i < args.warmup_epochs:
                pbar.set_postfix_str(f"loss: {loss.item():.2f}")
            else:
                pbar.set_postfix_str(
                    f"loss: {loss.item():.2f}, ce_loss: {ce_loss.item():.2f}, imitate_loss: {imitate_loss.item():.2f}"
                )
        # save the best imitation model
        if i > args.warmup_epochs:
            test_acc = get_acc(model, test_loader, device)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = copy.deepcopy(model)
        # Save a deepcopy of the model after warmup
        if i + 1 == args.warmup_epochs:
            warmup_model = copy.deepcopy(model)

        sched.step()

    return warmup_model, best_model


def imitate_in_train(model, pivot_dl, test_loader, device, args, epochs=20):
    """Continue training in model with pivot data"""
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    # Initialize loss functions
    ce_criterion = nn.CrossEntropyLoss()
    best_test_acc = 0
    best_model = None

    for i in range(epochs):
        loss_total = 0
        pbar = tqdm(pivot_dl)
        for itr, (x, y) in enumerate(pbar):
            if x.size(0) == 1:
                continue
            x, y = x.to(device), y.to(device)

            # Calculate CE loss
            loss = ce_criterion(model(x), y)
            loss_total += loss.item()
            pbar.set_postfix_str(f"loss: {loss.item():.2f}")
            optim.zero_grad()
            loss.backward()
            optim.step()

            test_acc = get_acc(model, test_loader, device)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = copy.deepcopy(model)
        sched.step()

    return best_model


def select_pivot_data(data_ds, teacher_model, args, device, strategy="random"):
    """Select pivot data from target dataset"""
    if strategy == "random":
        print(f"selecting pivot data randomly with balanced class distribution")
        labels = load_labels(data_ds)
        pivot_size = int(args.pkeep * len(data_ds))
        samples_per_class = pivot_size // args.num_classes

        pivot_indices = []
        for class_idx in range(args.num_classes):
            class_indices = np.where(labels == class_idx)[0]
            selected_indices = np.random.choice(class_indices, size=samples_per_class, replace=False)
            pivot_indices.extend(selected_indices)
        pivot_indices = np.array(pivot_indices)
    elif strategy == "balanced":
        # use balanced to select pivot data for instance proxy
        size = len(data_ds)
        np.random.seed(args.seed)
        keep_shadows = np.random.uniform(0, 1, size=(args.n_imitate_shadows, size))
        order_shadows = keep_shadows.argsort(0)
        keep_shadows = order_shadows < int(args.pkeep * size)
        keep = np.array(keep_shadows[args.shadow_id], dtype=bool)
        pivot_indices = keep.nonzero()[0]
    elif strategy == "low_loss":
        ce_criterion = nn.CrossEntropyLoss(reduction="none")
        data_dl = DataLoader(data_ds, batch_size=512, shuffle=False, num_workers=4)
        # Calculate k samples per class
        k = int(args.pkeep * len(data_ds) / args.num_classes)
        # Dictionary to store losses for each class
        class_losses = {i: [] for i in range(args.num_classes)}

        with torch.no_grad():
            batch_start_idx = 0
            for x, y in data_dl:
                x, y = x.to(device), y.to(device)
                logits = teacher_model(x)
                loss = ce_criterion(logits, y)

                # Store loss for each instance with its class
                for i in range(logits.size(0)):
                    instance_idx = batch_start_idx + i
                    class_idx = y[i].item()
                    class_losses[class_idx].append((instance_idx, loss[i].item()))
                batch_start_idx += x.size(0)

        # Select k lowest loss instances for each class
        pivot_indices = []
        for class_idx in range(args.num_classes):
            # Sort by loss and take k lowest
            class_losses[class_idx].sort(key=lambda x: x[1])
            selected_indices = [x[0] for x in class_losses[class_idx][:k]]
            pivot_indices.extend(selected_indices)
        pivot_indices = np.array(pivot_indices)
        print(f"selected {len(pivot_indices)} pivot data with low loss")
    else:
        raise ValueError(f"Strategy {strategy} not supported")

    return pivot_indices


def imitate_acc_rule(args):
    if args.dataset == "cifar10":
        args.imitate_acc = 0.6  
    elif args.dataset == "cifar100":
        args.imitate_acc = 0.4  
    elif args.dataset == "mnist" or args.dataset == "fmnist":
        args.imitate_acc = 0.8  
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

