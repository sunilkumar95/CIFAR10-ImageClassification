#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/main.py \
  --model_name="feed_forward" \
  --reset_output_dir \
  --data_path="cifar-10-batches-py" \
  --output_dir="outputs" \
  --log_every=100 \
  --num_epochs=10 \
  --train_batch_size=256 \
  --eval_batch_size=100 \
  --l2_reg=0.0001 \
  --learning_rate=0.05 \
  "$@"

