#!/usr/bin/env bash

modes=( probabilistic-drl reg-drl )
for aug in 'none'; do
  for mode in "${modes[@]}"; do
    for wf in 0 1 2; do
      sbatch --gres=gpu:rtx8000:1 --mem-per-cpu=8G -c 2 --ntasks=3 --time=95:59:00 --nodes=1 -J $mode run.sh cifar10 $aug $mode False $wf 0
      sbatch --gres=gpu:rtx8000:1 --mem-per-cpu=8G -c 2 --ntasks=3 --time=95:59:00 --nodes=1 -J $mode run.sh cifar100 $aug $mode False $wf 0
      sbatch --gres=gpu:rtx8000:1 --mem-per-cpu=8G -c 2 --ntasks=3 --time=95:59:00 --nodes=1 -J $mode run.sh mini_imgnet $aug $mode False $wf 0
    done
  done
done
