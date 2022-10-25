import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os

sns.color_palette("dark", as_cmap=True)
sns.set(style="darkgrid")

ss = 5
k = 10
times = (np.arange(k + 1) / (k))

datasets = ['cifar10', 'cifar100', 'mini_imgnet']
modes = ['probabilistic-drl', 'reg-drl']
wfs = [2, 4]

for dataset in datasets:
    for mode in modes:
        for wf in wfs:
            nmi = np.zeros([11, 11])
            for seed in range(3):
                mi = np.zeros([11, 11])
                hi = np.zeros([11, 11])
                hj = np.zeros([11, 11])

                for s in range(ss):
                    m = np.zeros([11, 11])
                    a = np.zeros([11, 11])
                    b = np.zeros([11, 11])

                    for idx in range(121):
                        if not os.path.exists(f'MI-Logs/{dataset}_{mode}_{wf}_{seed}_{s}_{idx}.pkl'):
                            # continue
                            print(f'Missing: {dataset} | {mode} | {wf} | {seed} | {s} | {idx}')
                            # os.system(
                            #    f"sbatch --gres=gpu:rtx8000 --mem=16G nmi.sh --data {dataset} --mode {mode} --seed {seed} --s {s} --wf {wf}")
                            continue

                        with open(f'MI-Logs/{dataset}_{mode}_{wf}_{seed}_{s}_{idx}.pkl', 'rb') as f:
                            data = pickle.load(f)

                        i = idx // len(times)
                        j = idx % len(times)

                        m[i,j] = data['MI']
                        a[i,j] = data['HI']
                        b[i,j] = data['HJ']

                        if np.isnan(m[i,j]).any() or np.isnan(a[i,j]).any() or np.isnan(b[i,j]).any():
                            print(f'NaN in: {dataset} | {mode} | {wf} | {seed} | {s} | {idx}')
                            # os.remove(f'MI-Logs/{dataset}_{mode}_{wf}_{seed}_{s}_{idx}.pkl')

                    mi += m / ss
                    hi += a / ss
                    hj += b / ss

                nmi += (mi / (np.sqrt(hi * hj + 1e-8))) / 3.

            ax = sns.heatmap(nmi, linewidths=0.5, cmap="YlGnBu",
                             xticklabels=times, yticklabels=times)
            print(nmi)
            plt.savefig(f'MI-Plots/nmi_{dataset}_{mode}_{wf}.pdf', bbox_inches='tight')
            plt.close()
