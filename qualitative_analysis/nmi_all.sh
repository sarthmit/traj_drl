#!/usr/bin/env bash

for data in cifar10 cifar100 mini_imgnet; do
  for mode in probabilistic-drl reg-drl; do
    for seed in 0 1 2; do
      for s in {0..5}; do
        for wf in 2 4; do
          sbatch --gres=gpu:rtx8000:1 --mem=16G --time=23:59:00 nmi.sh --data $data --mode $mode --seed $seed --s $s --wf $wf
        done
      done
    done
  done
done
