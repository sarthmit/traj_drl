import os

modes = ['probabilistic-drl', 'reg-drl']
seeds = range(10)
wfs = [0, 0.0625, 0.125, 0.25]
datasets = ['synthetic']
lrs = ['1e-4', '2e-4', '5e-4']

total = 0
pending = 0

for lr in lrs:
    for dataset in datasets:
        for mode in modes:
            for wf in wfs:
                avg_train = 0.
                avg_valid = 0.
                count = 0.
                for seed in range(10):
                    name = f'trained_models/{lr}/{mode}/{dataset}_{wf}_none_{seed}'
                    total += 1

                    if not os.path.exists(name):
                        pending += 1
                        print(name)
                        continue

                    with open(f'{name}/stdout.txt', 'r') as f:
                        data = f.read()

                    if 'step: 250000' not in data:
                        pending += 1
                        print(name)

                    data = data.split('\n')
                    count += 1
                    avg_train += float(data[-4].split(':')[-1])
                    avg_valid += float(data[-2].split(':')[-1])

                print(f'{lr} | {dataset} | {mode} | {wf} | {avg_train / count} | {avg_valid / count} | {count}')