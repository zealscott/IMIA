"""
This file is the script to run the CMIA framework (MIA shadow modeling part)
"""

import os
import argparse
from utils.loader import load_config

parser = argparse.ArgumentParser()

######### env configuration ########
parser.add_argument("--cuda", "-c", default=0, type=int)
parser.add_argument("--n_imitate_shadows", default=64, type=int)
parser.add_argument("--dataset", default="cifar10", type=str)
parser.add_argument("--temperature", default=1.0, type=float)

args = parser.parse_args()
load_config(args)
print(args)

for shadow_id in range(args.n_imitate_shadows):
    if shadow_id % args.n_gpus == args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        print(f"Training model {shadow_id}/{args.n_imitate_shadows} ...")
        cmd = f"python imitate_adapt_separate_mse.py --dataset {args.dataset} --shadow_id {shadow_id} --data_dir {args.data_dir} --savedir {args.savedir} --seed {args.seed} --model_type {args.model_type} --n_imitate_shadows {args.n_imitate_shadows} --temperature {args.temperature}"
        os.system(cmd)
