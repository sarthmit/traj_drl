#!/usr/bin/env bash

datasets=( cifar10 cifar100 mini_imgnet )
augs=( none )
modes=( probabilistic-drl reg-drl )
grans=( 1 2 5 7 10 17 25 37 50 )
lr='2e-4'

for data in "${datasets[@]}"; do
  for mode in "${modes[@]}"; do
    for aug in "${augs[@]}"; do
      for gran in "${grans[@]}"; do
        sbatch --gres=gpu:rtx8000:1 --mem-per-cpu=16G -c 2 --ntasks=1 --nodes=1 -J $data run_granular.sh $data $aug $mode $gran $lr
      done
    done
  done
done
