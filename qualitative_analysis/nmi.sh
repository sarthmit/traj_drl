#!/usr/bin/env bash
module load cuda/11.2/cudnn/8.1
module load anaconda/3
conda activate pytorch

for idx in {0..121}; do
  python nmi.py "$@" --idx $idx
done
