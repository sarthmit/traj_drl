#!/usr/bin/env bash

module load anaconda/3
module load cuda/11.0/cudnn/8.0
conda activate score

dataset=$1
aug=$2
extras=$3
mixup=$4
widen=$5
seed=$6
lr=2e-4

recon_method=l2
if [[ "$extras" == "probabilistic-drl" ]]; then
  prob=True
  lamb_z=1e-5
  recon=0
elif [[ "$extras" == "noreg-drl" ]]; then
  prob=False
  lamb_z=0
  recon=0
elif [[ "$extras" == "reg-drl" ]]; then
  prob=False
  lamb_z=1e-5
  recon=0
elif [[ "$extras" == "vae-l2" ]]; then
  prob=True
  lamb_z=1.
  recon=-1.
elif [[ "$extras" == "vae-bce" ]]; then
  prob=True
  lamb_z=1.
  recon=-1.
  recon_method=bce
elif [[ "$extras" == "ae-l2" ]]; then
  prob=False
  lamb_z=0.
  recon=-1.
elif [[ "$extras" == "ae-bce" ]]; then
  prob=False
  lamb_z=0.
  recon=-1.
  recon_method=bce
fi

if [[ "$mixup" == "True" ]]; then
  mix='_mixup'
else
  mix=''
fi

name="trained_models/"$lr"/"$extras"/"$dataset"_"$widen"_"$aug$mix
echo $name

srun -l python3 main.py --config=configs/ve/cifar10_ncsnpp_small_continuous.py \
--workdir=$name --mode=train --config.data.dataset=$dataset \
--config.training.experiment_name='' --config.training.include_encoder=True \
--config.training.probabilistic_encoder=$prob --config.training.lambda_z=$lamb_z \
--config.training.apply_mixup=$mixup --config.training.lambda_reconstr=$recon \
--config.training.n_iters=250000 --config.training.snapshot_freq=250000 \
--config.training.seed=$seed --config.model.widen_factor=$widen \
--config.training.recon=$recon_method --config.data.aug=$aug \
--config.optim.lr=$lr
