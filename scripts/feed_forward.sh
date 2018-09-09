#!/bin/bash

export PYTHONPATH="$(pwd)"

python src/main.py \
  --model_name="feed_forward" \
  --reset_output_dir \
  --data_path="cifar-10-batches-py" \
  --output_dir="outputs" \
  --batch_size=32 \
  --num_epochs=10 \
  --log_every=100 \
  --eval_every_epochs=1 \
  "$@"

