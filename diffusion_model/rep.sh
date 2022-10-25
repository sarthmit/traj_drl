#!/usr/bin/env bash

module load cuda/11.0/cudnn/8.0
module load anaconda/3
conda activate score

dataset=$1
wf=$2
lr=$3

augs=( none )
modes=( probabilistic-drl reg-drl )

for mode in "${modes[@]}"; do
  for aug in "${augs[@]}"; do
    for seed in {0..9}; do
      python3 representations.py --data $dataset --aug $aug --mode $mode --widen-factor $wf --seed $seed --lr $lr
    done
  done
done
