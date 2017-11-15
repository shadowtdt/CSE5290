#!/bin/bash
BOLD="\033[1m"
NORM="\033[0m"

# Input
OUTPUT_DIR=${1:-images}

# Common
REPO="local"
IMAGE="mnist_autoencoder"
TAG="1.0"

# CPU
CPU_FILE=Dockerfile.cpu
CPU_TAG="${TAG}-cpu"
CPU_IMAGE="${REPO}/${IMAGE}:${CPU_TAG}"
CPU_TAR="${OUTPUT_DIR}/${IMAGE}:${CPU_TAG}.tar"

# GPU
GPU_FILE=Dockerfile.gpu
GPU_TAG="${TAG}-gpu"
GPU_IMAGE="${REPO}/${IMAGE}:${GPU_TAG}"
GPU_TAR="${OUTPUT_DIR}/${IMAGE}:${GPU_TAG}.tar"

function buildImage()
{
file=$1
image=$2
echo -e "\n${BOLD}** Building ${image} Docker Image from file ${file} **${NORM}\n"
docker build -t ${image} -f ${file} .
}

function saveImage()
{
image=$1
output=$2
echo -e "\n${BOLD}** Saving ${image} Docker image to ${output} **${NORM}\n"
docker image save -o ${output} ${image}
}

echo -e "\n${BOLD}** Preparing Docker Context**${NORM}\n"
./prepareDockerContext.sh

buildImage ${CPU_FILE} ${CPU_IMAGE}
buildImage ${GPU_FILE} ${GPU_IMAGE}

#docker build -t /mnist_autoencoder:1.0-cpu -f Dockerfile.cpu .
#docker build -t local/mnist_autoencoder:1.0-gpu -f Dockerfile.gpu .

mkdir -p $OUTPUT_DIR
saveImage  ${CPU_IMAGE} ${CPU_TAR}
saveImage  ${GPU_IMAGE} ${GPU_TAR}
#docker image save -o ${OUTPUT_DIR}/${mnist_autoencoder_1.0-cpu}.tar ${local}${mnist_autoencoder}:${1.0-cpu}
#docker image save -o ${OUTPUT_DIR}/${mnist_autoencoder1.0-gpu}.tar ${local}${mnist_autoencoder}:${1.0-gpu}
