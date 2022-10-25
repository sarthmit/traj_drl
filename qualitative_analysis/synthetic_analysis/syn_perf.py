import os
import pickle
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description='Classifier on Representations')
parser.add_argument('--mode', type=str, default='probabilistic-drl',
                    help='Model mode')
parser.add_argument('--granularity', type=int, default=None)
parser.add_argument('--feat', type=str, default=None)
parser.add_argument('--score-lr', type=str, default=None)
args = parser.parse_args()

args.aug = 'none'
score_seeds = 10
downstream_seeds = 3

k = args.granularity
if args.granularity is None:
    ext_name = ''
else:
    ext_name = f'_{args.granularity}'
times = (np.arange(k + 1) / (k))
mappings = {
    'Location':
        {
            'num_classes': 16,
            't_idx': 2
        },
    'Background Color':
        {
            'num_classes': 9,
            't_idx': 0
        },
    'Foreground Color':
        {
            'num_classes': 9,
            't_idx': 1
        },
    'Object Shape':
        {
            'num_classes': 5,
            't_idx': 3
        }
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

device = torch.device('cuda:0')

down_lrs = [0.001, 0.00075, 0.0005, 0.00025, 0.0001, 0.00005]
criteria = np.median

for wf in [0., 0.0625, 0.125, 0.25]:
    if wf.is_integer():
        wf = int(wf)

    best_val = [0.]
    opt_test = [0.]

    for down_lr in down_lrs:
        val_accs = []
        test_accs = []
        for a in range(score_seeds):
            val_acc = 0.
            test_acc = 0.
            for b in range(downstream_seeds):
                if not os.path.exists(
                        f'synthetic_models/{args.score_lr}/{args.mode}/{wf}/{args.granularity}/{args.feat}/{down_lr}/{a}-{b}.pt'):
                    print(
                        f'synthetic_models/{args.score_lr}/{args.mode}/{wf}/{args.granularity}/{args.feat}/{down_lr}/{a}-{b}.pt')
                    print('Model Does Not Exist')
                    exit()

                val_acc += torch.load(f'synthetic_models/{args.score_lr}/{args.mode}/{wf}/{args.granularity}/{args.feat}/{down_lr}/{a}-{b}.pt', map_location=device)['Validation Accuracy'] / downstream_seeds
                test_acc += torch.load(f'synthetic_models/{args.score_lr}/{args.mode}/{wf}/{args.granularity}/{args.feat}/{down_lr}/{a}-{b}.pt', map_location=device)['Test Accuracy'] / downstream_seeds

            val_accs.append(val_acc)
            test_accs.append(test_acc)

        if criteria(val_accs) > criteria(best_val):
            best_val = val_accs
            opt_test = test_accs
            args.down_lr = down_lr

    print(f'Widen Factor: {wf} | Test Accuracy: {criteria(opt_test)}')