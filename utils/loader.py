import os
import pickle
import yaml

import numpy as np
import torch.nn as nn
from models.densenet import create_densenet121
from models.mobilenetv2 import create_mobilenetv2
from models.resnet import create_wideresnet
from models.vgg import create_vgg16
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torch.utils.data import Subset, ConcatDataset
import torch


def offline_data_split(dataset, seed=42, data_type="target"):
    """
    Split a dataset into target/shadow/reference sets with equal number of samples
    """
    total_size = len(dataset)
    all_indices = list(range(total_size))

    target_size = total_size // 3
    shadow_size = total_size // 3
    reference_size = total_size - target_size - shadow_size

    # Split into target and shadow
    target_indices, shadow_indices, reference_indices = torch.utils.data.random_split(
        all_indices, [target_size, shadow_size, reference_size], generator=torch.Generator().manual_seed(seed)
    )

    if data_type == "target":
        selected_indices = list(target_indices)
    elif data_type == "shadow":
        selected_indices = list(shadow_indices)
    elif data_type == "reference":
        selected_indices = list(reference_indices)
    else:
        raise ValueError(f"Invalid data type: {data_type}")

    selected_ds = Subset(dataset, selected_indices)

    print(f"loaded {len(selected_ds)} samples")

    return selected_ds


def load_dataset(args, data_type="target", augment=True):
    """
    load augmented datasets
    split the dataset into shadow/target/reference datasets (disjoint)
    """
    tv_dataset = get_dataset(args)

    # Decide transforms:
    # MNIST / fMNIST are grayscale 28x28; use the transforms for MNIST
    # CIFAR10 / CIFAR100 are 32x32 color; use the CIFAR-like transforms
    if args.dataset in ["mnist", "fmnist"]:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ]
        )
    if augment == False:
        train_transform = test_transform
    train_ds = tv_dataset(root=args.data_dir, train=True, download=True, transform=train_transform)
    test_ds = tv_dataset(root=args.data_dir, train=False, download=True, transform=test_transform)

    combined_ds = ConcatDataset([train_ds, test_ds])
    selected_ds = offline_data_split(combined_ds, seed=args.seed, data_type=data_type)

    return selected_ds


def load_labels(train_ds):
    """
    get the label w.r.t dataset
    """
    # Extract labels from the dataset
    if hasattr(train_ds, "targets"):
        # If dataset directly has targets attribute
        labels = train_ds.targets
    elif hasattr(train_ds.dataset, "targets"):
        # If it's a Subset, get targets from the underlying dataset
        indices = train_ds.indices
        labels = [train_ds.dataset.targets[i] for i in indices]
    else:
        # Fallback: manually extract labels from each sample
        labels = [train_ds[i][1] for i in range(len(train_ds))]

    return np.array(labels)


def get_dataset(args):
    """
    Returns the appropriate dataset class (not an instance) and sets
    dataset-specific mean, std, and num_classes in 'args'.
    """
    if args.dataset == "cifar10":
        args.data_mean = (0.4914, 0.4822, 0.4465)
        args.data_std = (0.2023, 0.1994, 0.2010)
        args.num_classes = 10

        return CIFAR10
    elif args.dataset == "cifar100":
        args.data_mean = (0.5071, 0.4867, 0.4408)
        args.data_std = (0.2675, 0.2565, 0.2761)
        args.num_classes = 100

        return CIFAR100
    elif args.dataset == "mnist":
        args.data_mean = (0.1307,)
        args.data_std = (0.3081,)
        args.num_classes = 10

        return MNIST
    elif args.dataset == "fmnist":
        # Example mean/std commonly used for FashionMNIST
        args.data_mean = (0.2860,)
        args.data_std = (0.3530,)
        args.num_classes = 10

        return FashionMNIST
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' not implemented.")


def load_model(args):
    """
    Load model with appropriate configurations based on dataset and model type.
    """
    if args.model_type == "resnet":
        m = create_wideresnet(dataset=args.dataset, num_classes=args.num_classes)
    elif args.model_type == "densenet":
        m = create_densenet121(dataset=args.dataset, num_classes=args.num_classes)
    elif args.model_type == "mobilenet":
        m = create_mobilenetv2(dataset=args.dataset, num_classes=args.num_classes)
    elif args.model_type == "vgg":
        m = create_vgg16(dataset=args.dataset, num_classes=args.num_classes)

    return m


def load_config(args):
    print(f"loading config for {args.dataset} ...")
    with open(f"config/{args.dataset}.yml", "r") as file:
        config = yaml.safe_load(file)
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    args.savedir = f"{args.dataset}/"
    os.makedirs(args.savedir, exist_ok=True)
