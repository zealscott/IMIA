# IMIA: Imitative Membership Inference Attack

This repository implements the **Imitative Membership Inference Attack (IMIA)**. It supports both adaptive and non-adaptive attack settings.

## Overview

IMIA introduces a new paradigm for membership inference attacks by training imitative models that mimic the behavior of target models. The attack operates in two main settings:

- **Adaptive Setting**: The adversary has access to the target model's potential training data
- **Non-Adaptive Setting**: The adversary's training data is strictly disjoint from the target model's training data

## Project Structure

```
├── attacks/                 # Attack implementations
│   ├── imiaadaptive.py     # Adaptive IMIA attack
│   ├── imianonadaptive.py  # Non-adaptive IMIA attack
│   └── ...                 # Baseline attacks (lira, rmia, etc.)
├── models/                  # Model architectures
│   ├── resnet.py
│   ├── vgg.py
│   ├── mobilenetv2.py
│   └── densenet.py
├── config/                  # Configuration files
│   ├── cifar10.yml
│   ├── cifar100.yml
│   └── ...
├── utils/                   # Utility functions
├── run_shadow.py           # Train shadow models
├── run_adapt_imitate_mse.py    # Train adaptive imitative models
├── run_nonadpt_imitate_mse.py  # Train non-adaptive imitative models
└── run_attack.py           # Run attacks
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

#### For Adaptive Setting:
```bash
python run_adapt_imitate_mse.py --dataset mnist --cuda 0 --n_imitate_shadows 5 --temperature 1.0
```

#### For Non-Adaptive Setting:
```bash
python run_nonadpt_imitate_mse.py --dataset mnist --cuda 0 --n_imitate_shadows 5 --temperature 1.0
```

### Step 3: Run Attacks

Execute the IMIA attacks:

```bash
# Run adaptive IMIA attack
python run_attack.py --dataset mnist --attack imiaadaptive

# Run non-adaptive IMIA attack
python run_attack.py --dataset mnist --attack imianonadaptive
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

**Note**: The used public datasets will be automatically downloaded to the specified `data_dir` if they do not exist.


## Baseline Attacks

All evaluated baselines are also included. It is easy to run baseline attacks (for example, LiRA) using the following command:

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
  - TPR@0.001%FPR
