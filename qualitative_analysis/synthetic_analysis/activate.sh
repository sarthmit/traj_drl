#!/usr/bin/env bash
module load cuda/11.2/cudnn/8.1
module load anaconda/3
conda activate pytorch

export PYTHONUNBUFFERED=1

for feat in "Object Shape" "Background Color" "Foreground Color" "Location"; do
  python syn_activations.py --feat "$feat" "$@"
done