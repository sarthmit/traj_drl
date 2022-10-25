import os
import argparse
import pickle
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

parser = argparse.ArgumentParser(description='Classifier on Representations')
parser.add_argument('--data', type=str, default='cifar10', choices=('cifar10', 'cifar100', 'mini_imgnet'))
parser.add_argument('--mode', type=str, default='probabilistic-drl', choices=('probabilistic-drl', 'reg-drl'))
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--s', type=int, default=0)
parser.add_argument('--score-lr', type=str, default='2e-4')
parser.add_argument('--wf', type=int, default=2)
parser.add_argument('--idx', type=int, default=0)
args = parser.parse_args()

class Mine(nn.Module):
    def __init__(self, wf=2, hidden_size=256):
        super().__init__()
        self.input_size = wf * 64
        self.fc1 = nn.Linear(self.input_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

    def mutual_information(self, joint, marginal, mine_net):
        t = mine_net(joint)
        et = torch.exp(mine_net(marginal))
        mi_lb = torch.mean(t) - torch.log(torch.mean(et))
        return mi_lb, t, et

    def learn_mine(self, batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
        joint, marginal = batch
        mi_lb, t, et = self.mutual_information(joint, marginal, mine_net)
        ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

        loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))

        mine_net_optim.zero_grad()
        autograd.backward(loss)
        mine_net_optim.step()
        return mi_lb, ma_et

    def sample_batch(self, data, batch_size=256, sample_mode='joint'):
        if sample_mode == 'joint':
            index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            batch = data[index]
        else:
            joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
            batch = torch.cat([data[joint_index][:,:self.input_size].view(-1, self.input_size),
                                             data[marginal_index][:, self.input_size:].view(-1, self.input_size)],
                                           dim=1)
        return batch

    def train(self, data, mine_net, mine_net_optim, batch_size=256, iter_num=int(5e+3), log_freq=int(1e+3)):
        result = list()
        ma_et = 1.
        for i in range(iter_num):
            batch = self.sample_batch(data, batch_size=batch_size) \
            , self.sample_batch(data, batch_size=batch_size, sample_mode='marginal')
            mi_lb, ma_et = self.learn_mine(batch, mine_net, mine_net_optim, ma_et)
            result.append(mi_lb.detach().cpu().numpy())
            if log_freq is not None:
                if (i + 1) % log_freq == 0:
                    print(result[-1])
        return self.ma(result)[-1]

    def ma(self, a, window_size=100):
        return [np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)]

k = 10
times = (np.arange(k + 1) / (k))
epochs = 100

aug = 'none'
ver = 'mine'
wf = args.wf
i = args.idx // len(times)
j = args.idx % len(times)

print(f'{i} | {j} | {args.s} | {args.seed}')
if os.path.exists(f'MI-Logs/{args.data}_{args.mode}_{wf}_{args.seed}_{args.s}_{args.idx}.pkl'):
    print('MI Log Exists')
    exit()

nm = f'../diffusion_model/trained_models/{args.score_lr}/{args.mode}/{args.data}_{wf}_{aug}_{args.seed}'
test_name = f'{nm}/test_10.pkl'
with open(test_name, 'rb') as f:
    test_data = pickle.load(f)

x_i = torch.Tensor(test_data['Representation'][times[i]])
x_j = torch.Tensor(test_data['Representation'][times[j]])

set_seed(args.s)
mine_net_indep = Mine(wf = args.wf).cuda()
mine_net_optim_indep = optim.Adam(mine_net_indep.parameters(), lr=1e-4)
x = torch.cat([x_i, x_j], dim = -1).cuda()
mi_estimate = mine_net_indep.train(x, mine_net_indep, mine_net_optim_indep, iter_num = epochs * (x_i.shape[0] // 256), log_freq = None)
print(f'MI: {mi_estimate}')

set_seed(args.s)
mine_net_indep = Mine(wf = args.wf).cuda()
mine_net_optim_indep = optim.Adam(mine_net_indep.parameters(), lr=1e-4)
x = torch.cat([x_i, x_i], dim = -1).cuda()
h_i = mine_net_indep.train(x, mine_net_indep, mine_net_optim_indep, iter_num = epochs * (x_i.shape[0] // 256), log_freq = None)
print(f'HI: {h_i}')

set_seed(args.s)
mine_net_indep = Mine(wf = args.wf).cuda()
mine_net_optim_indep = optim.Adam(mine_net_indep.parameters(), lr=1e-4)
x = torch.cat([x_j, x_j], dim = -1).cuda()
h_j = mine_net_indep.train(x, mine_net_indep, mine_net_optim_indep, iter_num = epochs * (x_i.shape[0] // 256), log_freq = None)
print(f'HJ: {h_j}')

data = {
    'MI': mi_estimate,
    'HI': h_i,
    'HJ': h_j
}

with open(f'MI-Logs/{args.data}_{args.mode}_{wf}_{args.seed}_{args.s}_{args.idx}.pkl', 'wb') as f:
    pickle.dump(data, f)