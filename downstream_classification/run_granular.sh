#!/usr/bin/env bash

module load cuda/11.0/cudnn/8.0
module load anaconda/3
conda activate pytorch

dataset=$1
aug=$2
extras=$3
gran=$4
lr=$5

echo $dataset
echo $aug
echo $extras
echo $gran
echo $lr

for wf in 1 2; do
  for seed in 0 1 2; do
    echo $wf
    echo $seed
    echo "MLP Model"
    python3 mlp_classification_hyp.py --data $dataset --aug $aug --mode $extras --widen-factor $wf --seed $seed --granularity $gran --score-lr $lr
    echo "RNN Model"
    python3 rnn_classification_hyp.py --data $dataset --aug $aug --mode $extras --widen-factor $wf --seed $seed --granularity $gran --score-lr $lr
    echo "Transformer Model"
    python3 transformer_classification_hyp.py --data $dataset --aug $aug --mode $extras --widen-factor $wf --seed $seed --granularity $gran --score-lr $lr
  done
done
