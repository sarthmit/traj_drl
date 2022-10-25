#!/usr/bin/env bash

datasets=( synthetic cmnist )

for data in "${datasets[@]}"; do
  for wf in 0 0.0625 0.125 0.25; do
    for lr in 1e-4 2e-4 5e-4; do
      sbatch --gres=gpu:rtx8000:1 --mem=16G -J "rep_"$data rep.sh $data $wf $lr
    done
  done
done

datasets=( cifar10 cifar100 mini_imgnet )

for data in "${datasets[@]}"; do
    for wf in 1 2; do
      sbatch --gres=gpu:rtx8000:1 --mem=16G -J "rep_"$data rep.sh $data $wf
    done
done
