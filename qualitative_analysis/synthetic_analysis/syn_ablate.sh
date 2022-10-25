#!/usr/bin/env bash

#for wf in 0 0.0625 0.125 0.25 0.5; do
for wf in 0 0.0625 0.125 0.25 0.5; do
  for granularity in 10; do
    for score_lr in 1e-4 2e-4 5e-4; do
      sbatch --gres=gpu:rtx8000:1 --mem=16G -c 2 --ntasks=6 sablate.sh --widen-factor $wf --mode reg-drl --granularity $granularity --score-lr $score_lr "$@"
      sbatch --gres=gpu:rtx8000:1 --mem=16G -c 2 --ntasks=6 sablate.sh --widen-factor $wf --mode probabilistic-drl --granularity $granularity --score-lr $score_lr"$@"
    done
  done
done