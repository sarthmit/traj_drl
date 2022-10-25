import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy

pd.set_option('display.max_rows', None)
sns.color_palette("dark", as_cmap=True)
sns.set(style="darkgrid")

parser = argparse.ArgumentParser(description='Classifier on Representations')
parser.add_argument('--mode', type=str, default='probabilistic-drl',
                    help='Model mode')
parser.add_argument('--widen-factor', default=2., type=float, metavar='N',
                    help='widen factor for WReN')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--score-lr', type=str, default='2e-4')
parser.add_argument('--down-lr', type=float, default=1e-3)
parser.add_argument('--granularity', type=int, default=None)
args = parser.parse_args()

lrs = [0.001, 0.00075, 0.0005, 0.00025, 0.0001, 0.00005]
lr_index = int(os.environ['SLURM_PROCID'])
args.down_lr = lrs[lr_index]

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

if args.widen_factor.is_integer():
    args.widen_factor = int(args.widen_factor)

in_dim = int(args.widen_factor * 64)
if in_dim == 0:
    in_dim = 2

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

df = pd.DataFrame(columns=['Labels', 'Time', 'Attention Score'])
device = torch.device('cuda:0')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len = 75):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class Self_Attention(nn.Module):
    def __init__(self, dim, nheads=4, dropout=0.0):
        super(Self_Attention, self).__init__()

        self.dim = dim
        self.nheads = nheads
        self.head_dim = dim // nheads

        self.norm_before = False

        self.query_net = nn.Linear(dim, dim)
        self.key_net = nn.Linear(dim, dim)
        self.value_net = nn.Linear(dim, dim)

        self.final = nn.Linear(dim, dim)

        self.res = nn.Sequential(
            nn.Linear(dim,2 * dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.Dropout(p=dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        bsz, n_read, _ = x.shape
        _, n_write, _ = x.shape

        res = x
        if self.norm_before:
            x = self.norm1(x)

        q = self.query_net(x).reshape(bsz, n_read, self.nheads, self.head_dim)
        q = q.permute(0,2,1,3) / np.sqrt(self.head_dim)
        k = self.key_net(x).reshape(bsz, n_write, self.nheads, self.head_dim)
        k = k.permute(0,2,3,1)
        v = self.value_net(x).reshape(bsz, n_write, self.nheads, self.head_dim)
        v = v.permute(0,2,1,3)

        score = torch.matmul(q, k)
        mask = torch.zeros_like(score).detach()
        mask[:, :, :, -1] = 1.
        mask = mask.bool()
        score.masked_fill_(mask, float('-inf'))
        score = F.softmax(score, dim=-1)

        out = torch.matmul(score, v) # (bsz, nheads, n_read, att_dim)
        out = out.view(bsz, self.nheads, n_read, self.head_dim)

        out = out.permute(0, 2, 1, 3).reshape(bsz, n_read, self.dim)
        out = self.final(out)

        if not self.norm_before:
            out = self.norm1(res + out)
        else:
            out = res + out

        res = out

        if self.norm_before:
            out = self.norm2(out)
            out = res + self.res(out)
        else:
            out = self.norm2(res + self.res(out))

        return out, score

class Model(nn.Module):
    def __init__(self, in_dim=128, num_classes=10, nheads=4, iters=1):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.nheads = nheads
        self.iters = iters
        self.pe = PositionalEncoding(d_model=256)
        drop = 0.25

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Dropout(drop)
        )

        self.model = Self_Attention(256, nheads, dropout=drop)
        self.embedding = nn.Parameter(0.1 * torch.randn(1, 1, 256))
        nn.init.xavier_uniform_(self.embedding)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pe(x)
        x = torch.cat([x, self.embedding.repeat(x.shape[0], 1, 1)], dim=1)
        for _ in range(self.iters):
            x, score = self.model(x)

        return self.out(x[:, -1, :]), score[:,:,-1,:]

def train(epoch, loader, feat):
    model.train()
    total_loss = 0.
    total_acc = 0.
    total = 0.
    with tqdm(loader, unit="batch", ncols=100, disable=True) as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}")
            data, target = data.to(device), target.long()[:, mappings[feat]['t_idx']].to(device)
            optimizer.zero_grad()
            output, _ = model(data)

            loss = criterion(output, target)
            prediction = output.argmax(dim=1, keepdim=True).squeeze()
            correct = (prediction == target).sum().item()

            total_loss += loss.item()
            total_acc += correct
            total += data.shape[0]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tepoch.set_postfix(acc=100. * (total_acc / total),
                               feature=feat)

    if epoch % 100 == 0:
        print(f'Feature: {feat} | Training Epoch: {epoch} | Accuracy: {100. * (total_acc / total)}')
    return 100. * (total_acc / total)

def eval(epoch, loader, feat, compute=False):
    model.eval()
    total_loss = 0.
    total_acc = 0.
    total = 0.

    with torch.no_grad():
        with tqdm(loader, unit="batch", ncols=100, disable=True) as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Eval Epoch {epoch}")
                data, target = data.to(device), target.long()[:, mappings[feat]['t_idx']].to(device)
                output, _ = model(data)

                prediction = output.argmax(dim=1, keepdim=True).squeeze()
                correct = (prediction == target).sum().item()

                total_acc += correct
                total += data.shape[0]

                tepoch.set_postfix(acc=100. * (total_acc / total),
                                   feature=feat)

    if epoch % 100 == 0:
        if not compute:
            print(f'Feature: {feat} | Validation Epoch: {epoch} | Accuracy: {100. * (total_acc / total)}')
        else:
            print(f'Feature: {feat} | Test Epoch: {epoch} | Accuracy: {100. * (total_acc / total)}')
    return 100. * (total_acc / total)

for s in range(score_seeds):
    nm = f'../../diffusion_model/trained_models/{args.score_lr}/{args.mode}/synthetic_{args.widen_factor}_{args.aug}_{s}'

    train_name = f'{nm}/train{ext_name}.pkl'
    test_name = f'{nm}/test{ext_name}.pkl'

    with open(train_name, 'rb') as f:
        train_data = pickle.load(f)

    with open(test_name, 'rb') as f:
        test_data = pickle.load(f)

    for seed in range(downstream_seeds):
        set_seed(seed)
        idx = np.arange(4000)
        np.random.shuffle(idx)

        train_idx = idx[:3500]
        val_idx = idx[3500:]

        X_train = torch.Tensor(train_data['Representation'][0]).unsqueeze(1)
        y_train = torch.Tensor(train_data['Labels']).view(-1, 4)

        X_test = torch.Tensor(test_data['Representation'][0]).unsqueeze(1)
        y_test = torch.Tensor(test_data['Labels']).view(-1, 4)

        for time in times[1:]:
            X_train = torch.cat([X_train, torch.Tensor(train_data['Representation'][time]).unsqueeze(1)], dim=1)
            X_test = torch.cat([X_test, torch.Tensor(test_data['Representation'][time]).unsqueeze(1)], dim=1)

        X_val = X_train[val_idx]
        y_val = y_train[val_idx]

        X_train = X_train[train_idx]
        y_train = y_train[train_idx]

        train_dataset = TensorDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        val_dataset = TensorDataset(X_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        test_dataset = TensorDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        del X_train, X_val, X_test, y_train, y_val, y_test

        for k, feat in enumerate(sorted(mappings.keys())):
            if os.path.exists(f'synthetic_models/{args.score_lr}/{args.mode}/{args.widen_factor}/{args.granularity}/{feat}/{args.down_lr}/{s}-{seed}.pt'):
                continue

            model = Model(in_dim, mappings[feat]['num_classes']).to(device)
            num_params = sum(p.numel() for p in model.parameters())
            if seed == 0 and s == 0:
                print(model)
                print(f"Number of Parameters: {num_params}")

            optimizer = torch.optim.Adam(model.parameters(), lr=args.down_lr, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            best_val = 0.

            for epoch in range(1, args.epochs + 1):
                train_acc = train(epoch, train_dataloader, feat)
                val_acc = eval(epoch, val_dataloader, feat)

                if val_acc > best_val:
                    best_val = val_acc
                    opt_test = eval(epoch, test_dataloader, feat, True)
                    opt_model = deepcopy(model.state_dict())

            save_dict = {
                'Train Accuracy': train_acc,
                'Validation Accuracy': best_val,
                'Test Accuracy': opt_test,
                'Learning Rate': args.down_lr,
                'Score Seed': s,
                'Downstream Seed': seed,
                'state_dict': opt_model
            }

            if not os.path.exists(
                    f'synthetic_models/{args.score_lr}/{args.mode}/{args.widen_factor}/{args.granularity}/{feat}/{args.down_lr}/'):
                os.makedirs(
                    f'synthetic_models/{args.score_lr}/{args.mode}/{args.widen_factor}/{args.granularity}/{feat}/{args.down_lr}/')

            torch.save(save_dict,
                       f'synthetic_models/{args.score_lr}/{args.mode}/{args.widen_factor}/{args.granularity}/{feat}/{args.down_lr}/{s}-{seed}.pt')
            del model

        del train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader