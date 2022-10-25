import argparse
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Classifier on Representations')
parser.add_argument('--mode', type=str, default='probabilistic-drl',
                    help='Model mode')
parser.add_argument('--widen-factor', default=2., type=float, metavar='N',
                    help='widen factor for WReN')
parser.add_argument('--granularity', type=int, default=None)
args = parser.parse_args()

if args.widen_factor.is_integer():
    args.widen_factor = int(args.widen_factor)

jsd_ = np.zeros((40, 40))

def kld(p_a, p_b):
    return (p_a * np.log2(p_a) - p_a * np.log2(p_b)).sum().item()

def jsd(p_a, p_b):
    p_ab = (p_a + p_b) / 2.
    return 0.5 * (kld(p_a, p_ab) + kld(p_b, p_ab))

for feat_a in range(40):
    act_a = torch.load(f'celeba_activations/{args.mode}/{args.widen_factor}/{args.granularity}/{feat_a}.pt')
    act_a = act_a.view(3, -1, 11, 2).mean(dim=0).mean(dim=0)[:, 1]
    for feat_b in range(40):
        act_b = torch.load(f'celeba_activations/{args.mode}/{args.widen_factor}/{args.granularity}/{feat_b}.pt')

        act_b = act_b.view(3, -1, 11, 2).mean(dim=0).mean(dim=0)[:,1]

        jsd_[feat_a, feat_b] = jsd(act_a, act_b)
        print(f'Feature {feat_a} | Feature {feat_b} | JSD: {jsd_[feat_a, feat_b]}')

mask = np.zeros_like(jsd_)
mask[np.tril_indices_from(mask)] = True

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(16, 8))
    ax = sns.heatmap(jsd_, mask=mask, square=True, cmap="YlGnBu", linewidths=.25)

plt.savefig('test.png')
print(jsd_)
