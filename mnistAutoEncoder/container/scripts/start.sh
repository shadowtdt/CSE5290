#!/bin/sh
BOLD="\033[1m"
NORM="\033[0m"
echo -e "Starting MNIST Autoencoder"
echo -e "Model Args: ${MODEL_ARGS}"

echo -e "${BOLD}\n** Note GPU image will fail if Nivida CUDA files are not present https://github.com/tensorflow/tensorflow/issues/4078 **\n${NORM}"

python /home/mnist_autoEncoder.py --log_dir="${LOG_DIR}" ${MODEL_ARGS}
tensorboard --logdir=${LOG_DIR}
