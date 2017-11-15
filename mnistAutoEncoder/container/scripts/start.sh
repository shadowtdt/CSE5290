#!/bin/sh

python /home/mnist_autoEncoder.py --log_dir="${LOG_DIR}" ${MODEL_ARGS}
tensorboard --logdir=${LOG_DIR}

read -p "Execution has finished. View results via Tensorboard at http://localhost:6006. Press any key to terminate container"