# IMIA: Imitative Membership Inference Attack

This repository implements the **Imitative Membership Inference Attack (IMIA)**. It supports both adaptive and non-adaptive attack settings.

## Overview

IMIA introduces a new paradigm for membership inference attacks by training imitative models that mimic the behavior of target models. The attack operates in two main settings:

- **Adaptive Setting**: The adversary has access to the membership queries, which includes target model's training data.
- **Non-Adaptive Setting**: The adversary's data is strictly disjoint from the target model's training data.

## Project Structure

```
├── attacks/                    # Attack implementations
│   ├── imiaadaptive.py         # Adaptive IMIA attack
│   ├── imianonadaptive.py      # Non-adaptive IMIA attack
│   └── ...                     # Baseline attacks (lira, rmia, etc.)
├── models/                     # Model architectures
│   ├── resnet.py
│   ├── vgg.py
│   ├── mobilenetv2.py
│   └── densenet.py
├── config/                      # Configuration files
│   ├── cifar10.yml
│   ├── cifar100.yml
│   └── ...
├── utils/                      # Utility functions
├── run_shadow.py               # Train shadow models
├── run_adaptive_imia.py        # Train adaptive imitative models
├── run_nonadaptive_imia.py     # Train non-adaptive imitative models
├── imitate_adaptive_imia.py    # Adaptive IMIA training script
├── imitate_nonadaptive_imia.py # Non-adaptive IMIA training script
└── run_attack.py               # Run attacks
```

## Setup

### Prerequisites

- Python 3.7+
- PyTorch
- Required packages: `numpy`, `scipy`, `scikit-learn`, `faiss-cpu`, `clip`


## Usage

The IMIA framework consists of three main steps:

### Step 1: Train Shadow Models

First, train shadow models (default: 256 models) used for baselines, and the last one is the target model by default:

```bash
python run_shadow.py --dataset mnist --cuda 0
```

### Step 2: Train Imitative Models

The IMIA framework trains imitative models that mimic the behavior of target models. Hyperparameters (margin_weight, warmup_epochs, temperature) are automatically loaded from the dataset-specific configuration files in the `config/` directory.

For Adaptive Setting:
```bash
python run_adaptive_imia.py --dataset mnist --cuda 0 --n_imitate_shadows 10
```

For Non-Adaptive Setting:
```bash
python run_nonadaptive_imia.py --dataset mnist --cuda 0 --n_imitate_shadows 10
```

**Parameters:**
- `--dataset`: Dataset name (mnist, fmnist, cifar10, cifar100)
- `--cuda`: GPU ID to use
- `--n_imitate_shadows`: Number of imitative shadow models to train (default: 64)

### Step 3: Run Attacks

Execute the IMIA attacks:

```bash
# Run non-adaptive IMIA attack
python run_attack.py --dataset mnist --attack imianonadaptive

# Run adaptive IMIA attack
python run_attack.py --dataset mnist --attack imiaadaptive
```

**Parameters:**
- `--dataset`: Dataset name
- `--attack`: Attack method (`imiaadaptive`, `imianonadaptive`, or baseline methods)

## Configuration

Edit the configuration files in the `config/` directory to customize:

- Number of shadow models (`n_shadows`)
- Model architecture (`model_type`)
- Data directory (`data_dir`)
- Number of GPUs (`n_gpus`)
- Random seed (`seed`)
- IMIA hyperparameters:
  - `margin_weight`: Weight for margin-focused imitation
  - `warmup_epochs`: Number of warmup epochs before imitation 
  - `temperature`: Temperature scaling for imitation
  - `n_query`: Number of queries for each instance

The used public datasets will be automatically downloaded to the specified `data_dir` if they do not exist.

## Data Splits

We split each dataset into three disjoint parts: **target**, **shadow**, and **reference**. The split is performed on the combined train and test sets of each dataset, ensuring complete separation between different data types.

### Dataset Split Sizes

| Dataset | Total Samples | Target Split | Shadow Split | Reference Split |
|---------|---------------|--------------|--------------|-----------------|
| MNIST | 70,000 | 23,333 | 23,333 | 23,334 |
| FashionMNIST | 70,000 | 23,333 | 23,333 | 23,334 |
| CIFAR-10 | 60,000 | 20,000 | 20,000 | 20,000 |
| CIFAR-100 | 60,000 | 20,000 | 20,000 | 20,000 |

The detailed implementation is in `utils/loader.py`.

### Data Usage by Setting

| Component | Adaptive Setting | Non-Adaptive Setting |
|-----------|------------------|---------------------|
| **Target Model Training** | Shadow split | Shadow split |
| **Shadow Models Training** | Shadow split | Target split |
| **Imitative Models Training** | Shadow split | Target split |

The split ensures that:
- Target and shadow data are completely disjoint
- Reference data is separate and can be used for other purposes (e.g., baseline attacks)
- Each split maintains the original class distribution of the dataset

## Baseline Attacks

After training shadow models, it is easy to run baseline attacks (for example, LiRA) using the following command:

```bash
python run_attack.py --dataset mnist --attack lira
```

## Attack Evaluation

The framework evaluates attacks using multiple metrics:

- **AUC**: Area Under the ROC Curve
- **ACC**: Accuracy
- **TPR@X%FPR**: True Positive Rate at X% False Positive Rate
  - TPR@10%FPR
  - TPR@1%FPR
  - TPR@0.1%FPR
  - TPR@0.01%FPR
  - TPR@0.001%FPR (equivalent to TPR@0%FPR for evaluated datasets)

## Acknowledgements

Our implementation is built upon the following open-source repositories:
- [CMIA & PMIA](https://github.com/zealscott/MIA)
- [RAPID](https://github.com/T0hsakar1n/RAPID)
- [LiRA-Pytorch](https://github.com/orientino/lira-pytorch/)


We sincerely appreciate their valuable contributions to the community.
