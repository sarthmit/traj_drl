#!/usr/bin/env bash

for wf in 2; do
  for granularity in 2 5; do
    for score_lr in 2e-4; do
      for down_lr in 0.001 0.00075 0.0005 0.00025 0.0001 0.00005; do
        sbatch --gres=gpu:rtx8000:1 --time=120:0:0 --mem=16G -c 1 celebate.sh --widen-factor $wf --mode reg-drl --granularity $granularity --score-lr $score_lr --down-lr $down_lr "$@"
        sbatch --gres=gpu:rtx8000:1 --time=120:0:0 --mem=16G -c 1 celebate.sh --widen-factor $wf --mode probabilistic-drl --granularity $granularity --score-lr $score_lr --down-lr $down_lr "$@"
      done
    done
  done
done
