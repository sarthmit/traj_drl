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
parser.add_argument('--widen-factor', default=2., type=float, metavar='N',
                    help='widen factor for WReN')
parser.add_argument('--granularity', type=int, default=None)
parser.add_argument('--feat', type=int, default=0)
args = parser.parse_args()

args.aug = 'none'
score_seeds = 3
downstream_seeds = 3

k = args.granularity
if args.granularity is None:
    ext_name = ''
else:
    ext_name = f'_{args.granularity}'
times = (np.arange(k + 1) / (k))

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

def eval(epoch, loader, feat):
    model.eval()
    total_acc = 0.
    total = 0.
    att_scores = torch.Tensor([])

    with torch.no_grad():
        with tqdm(loader, unit="batch", ncols=100, disable=True) as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Eval Epoch {epoch}")
                data, target = data.to(device), target[:, feat].to(device)
                output, score = model(data)
                att_scores = torch.cat([att_scores, score[:, :, :-1].cpu()], dim=0)
                output = output.squeeze()

                prediction = (output >= 0.).long()
                correct = (prediction == target).sum().item()

                total_acc += correct
                total += data.shape[0]

                tepoch.set_postfix(acc=100. * (total_acc / total),
                                   feature=feat)

    tms = torch.Tensor(times).view(1, 1, -1).repeat(att_scores.shape[0], att_scores.shape[1], 1)
    final = torch.cat([tms.view(-1, 1), att_scores.view(-1, 1)], dim=-1)
    print(f'Feature: {feat} | Test Epoch: {epoch} | Accuracy: {100. * (total_acc / total)}')
    return 100. * (total_acc / total), final

score_lrs = ['2e-4']
down_lrs = [0.001, 0.00075, 0.0005, 0.00025, 0.0001, 0.00005]
best_val = 0.

for score_lr in score_lrs:
    for down_lr in down_lrs:
        val_acc = 0.
        for a in range(score_seeds):
            for b in range(downstream_seeds):
                if not os.path.exists(f'celeba_models/{score_lr}/{args.mode}/{args.widen_factor}/{args.granularity}/{args.feat}/{down_lr}/{a}-{b}.pt'):
                    print(f'celeba_models/{score_lr}/{args.mode}/{args.widen_factor}/{args.granularity}/{args.feat}/{down_lr}/{a}-{b}.pt')
                    print('Model Does Not Exist')
                    exit()

                val_acc += torch.load(f'celeba_models/{score_lr}/{args.mode}/{args.widen_factor}/{args.granularity}/{args.feat}/{down_lr}/{a}-{b}.pt', map_location=device)['Validation Accuracy'] / (score_seeds * downstream_seeds)

        if val_acc > best_val:
            best_val = val_acc
            args.score_lr = score_lr
            args.down_lr = down_lr

activations = []
print(f'Optimal Downstream Learning Rate: {args.down_lr}')
for s in range(score_seeds):
    nm = f'../../score_sde_pytorch/trained_models/{args.score_lr}/{args.mode}/celeba_{args.widen_factor}_{args.aug}_{s}'
    test_name = f'{nm}/test{ext_name}.pkl'

    with open(test_name, 'rb') as f:
        test_data = pickle.load(f)

    X_test = torch.Tensor(test_data['Representation'][0]).unsqueeze(1)
    y_test = torch.Tensor(test_data['Labels']).view(-1, 40)

    for time in times[1:]:
        X_test = torch.cat([X_test, torch.Tensor(test_data['Representation'][time]).unsqueeze(1)], dim=1)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=250, shuffle=False, drop_last=False)

    del X_test, y_test

    acts = []
    for seed in range(downstream_seeds):
        save_dict = torch.load(f'celeba_models/{args.score_lr}/{args.mode}/{args.widen_factor}/{args.granularity}/{args.feat}/{args.down_lr}/{s}-{seed}.pt', map_location=device)
        state_dict = save_dict['state_dict']
        opt_test = save_dict['Test Accuracy']

        model = Model(in_dim, 1).to(device)
        print(f'Pre-Load Test Accuracy: {opt_test}')
        model.load_state_dict(state_dict)
        _, act = eval(0, test_dataloader, args.feat)
        acts.append(act.numpy())

    acts = np.array(acts)
    activations.append(torch.Tensor(acts).mean(dim=0).numpy())

if not os.path.exists(f'celeba_activations/{args.mode}/{args.widen_factor}/{args.granularity}/'):
    os.makedirs(f'celeba_activations/{args.mode}/{args.widen_factor}/{args.granularity}')

torch.save(torch.Tensor(np.array(activations)), f'celeba_activations/{args.mode}/{args.widen_factor}/{args.granularity}/{args.feat}.pt')