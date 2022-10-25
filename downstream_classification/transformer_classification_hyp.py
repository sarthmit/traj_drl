import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser(description='Classifier on Representations')
parser.add_argument('--data', type=str, default='cifar10',
                    help='path to dataset')
parser.add_argument('--mode', type=str, default='probabilistic-drl',
                    help='Model mode')
parser.add_argument('--aug', type=str, default='none',
                    help='Aug mode')
parser.add_argument('--widen-factor', default=2, type=int, metavar='N',
                    help='widen factor for WReN')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--granularity', type=int, default=None)
parser.add_argument('--score-lr', type=str, default='2e-4')
args = parser.parse_args()

args.seed += int(os.environ['SLURM_PROCID'])
k = args.granularity
if args.granularity is None:
    ext_name = ''
else:
    ext_name = f'_{args.granularity}'

if k == 1:
    times = [0.5]
else:
    times = (np.arange(k + 1) / (k))
seeds = 5
lrs = [0.001, 0.00075, 0.0005, 0.00025, 0.0001, 0.00005]

if args.data == 'cifar10':
    num_classes = 10
else:
    num_classes = 100

if args.widen_factor == 2:
    in_dim = 128
else:
    in_dim = 256

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

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

        score = F.softmax(torch.matmul(q,k), dim=-1) # (bsz, nheads, n_read, n_write)

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

        return out

class Transformer(nn.Module):
    def __init__(self, in_dim=128, num_classes=10, dim=256, nheads=4, iters=2):
        super(Transformer, self).__init__()
        self.in_dim = in_dim
        self.nheads = nheads
        self.iters = iters
        self.pe = PositionalEncoding(d_model=dim)
        drop = 0.25

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.Dropout(drop)
        )

        self.model = Self_Attention(dim, nheads, dropout=drop)
        self.embedding = nn.Parameter(0.1 * torch.randn(1, 1, dim))
        nn.init.xavier_uniform_(self.embedding)
        self.out = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pe(x)
        x = torch.cat([x, self.embedding.repeat(x.shape[0], 1, 1)], dim=1)
        scores = []
        for _ in range(self.iters):
            x = self.model(x)

        return self.out(x[:, -1, :])

def train(epoch, loader):
    model.train()
    total_loss = 0.
    total_acc = 0.
    total = 0.
    with tqdm(loader, unit="batch", ncols=90) as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Train Epoch {epoch}")
            data, target = data.to(device), target.long().to(device)
            optimizer.zero_grad()
            output = model(data)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            loss = criterion(output, target)
            correct = (predictions == target).sum().item()

            total_loss += loss.item()
            total_acc += correct
            total += data.shape[0]

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tepoch.set_postfix(loss=total_loss / total, accuracy=100. * (total_acc / total))

    return 100. * (total_acc / total)

def eval(epoch, loader):
    model.eval()
    total_loss = 0.
    total_acc = 0.
    total = 0.

    with torch.no_grad():
        with tqdm(loader, unit="batch", ncols=90) as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Test Epoch {epoch}")
                data, target = data.to(device), target.long().to(device)
                output = model(data)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = criterion(output, target)
                correct = (predictions == target).sum().item()

                total_loss += loss.item()
                total_acc += correct
                total += data.shape[0]

                tepoch.set_postfix(loss=total_loss / total, accuracy=100. * (total_acc / total))

    return 100. * (total_acc / total)

nm = f'../diffusion_model/trained_models/{args.score_lr}/{args.mode}/{args.data}_{args.widen_factor}_{args.aug}_{args.seed}'

if os.path.exists(f'{nm}/eval_log_tsf_opt{ext_name}.txt'):
    print('Evaluation Already Done')
    exit()

if args.granularity == 1:
    train_name = f'{nm}/train_2.pkl'
    test_name = f'{nm}/test_2.pkl'
else:
    train_name = f'{nm}/train{ext_name}.pkl'
    test_name = f'{nm}/test{ext_name}.pkl'

with open(train_name, 'rb') as f:
    train_data = pickle.load(f)

with open(test_name, 'rb') as f:
    test_data = pickle.load(f)

vals = np.zeros([len(lrs), seeds])
perf_trains = np.zeros([len(lrs), seeds])
perf_tests = np.zeros([len(lrs), seeds])

for seed in range(seeds):
    set_seed(seed)
    idx = np.arange(50000)
    np.random.shuffle(idx)

    train_idx = idx[:45000]
    val_idx = idx[45000:]

    if args.granularity == 1:
        X_train = torch.Tensor(train_data['Representation'][0.5]).unsqueeze(1)
        X_test = torch.Tensor(test_data['Representation'][0.5]).unsqueeze(1)
    else:
        X_train = torch.Tensor(train_data['Representation'][0]).unsqueeze(1)
        X_test = torch.Tensor(test_data['Representation'][0]).unsqueeze(1)

        for time in times[1:]:
            X_train = torch.cat([X_train, torch.Tensor(train_data['Representation'][time]).unsqueeze(1)], dim=1)
            X_test = torch.cat([X_test, torch.Tensor(test_data['Representation'][time]).unsqueeze(1)], dim=1)

    y_train = torch.Tensor(train_data['Labels'])
    y_test = torch.Tensor(test_data['Labels'])

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    train_dataset = TensorDataset(X_train, y_train)
    del X_train, y_train
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_dataset = TensorDataset(X_val, y_val)
    del X_val, y_val
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    test_dataset = TensorDataset(X_test, y_test)
    del X_test, y_test
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    for i, lr in enumerate(lrs):
        model = Transformer(in_dim, num_classes).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        if seed == 0 and i == 0:
            print(model)
            print(f"Number of Parameters: {num_params}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        best_val = 0.
        opt_test = 0.

        for epoch in range(1, args.epochs + 1):
            train_acc = train(epoch, train_dataloader)
            val_acc = eval(epoch, val_dataloader)
            test_acc = eval(epoch, test_dataloader)

            if val_acc > best_val:
                best_val = val_acc
                opt_test = test_acc

        vals[i, seed] = best_val
        perf_trains[i, seed] = train_acc
        perf_tests[i, seed] = opt_test

vals = np.mean(vals, axis=1)
idx = np.argmax(vals)
perf_train = np.mean(perf_trains, axis=1)[idx]
perf_test = np.mean(perf_tests, axis=1)[idx]
lr = lrs[idx]

print(np.mean(perf_tests, axis=1))
log = 'Pretrained:\n'
log += f'Learning Rate: {lr} | Train Accuracy: {perf_train} | Test Accuracy: {perf_test}\n'

with open(f'{nm}/eval_log_tsf_opt{ext_name}.txt', 'w') as f:
    f.write(log)
print(log)