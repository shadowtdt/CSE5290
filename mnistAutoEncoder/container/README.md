# Using MNIST Autoencoder wiht Docker

This directory contains `Dockerfile`s that define the images

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/)


## Which containers exist?

We currently maintain two Docker container images:

* `local/autoencoder:1.0-cpu` - TensorFlow with all dependencies - CPU only!

* `loca/autoencoder:1.0-gpu` - - TensorFlow with all dependencies and support for NVidia CUDA

## Running the container

Run non-GPU container using

    $ docker run -it -p 6006:6006 local/autoencoder:1.0-cpu

For GPU support install NVidia drivers (ideally latest) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Run using

    $ nvidia-docker run -it -p 6006:6006 local/autoencoder:1.0-gpu


Note: If you would have a problem running nvidia-docker you may try the old method
we have used. But it is not recommended. If you find a bug in nvidia-docker, please report
it there and try using nvidia-docker as described above.

    $ # The old, not recommended way to run docker with gpu support: 
    $ export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | xargs -I{} echo '-v {}:{}')
    $ export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    $ docker run -it -p 6006:6006 $CUDA_SO $DEVICES local/autoencoder:1.0-

If you would like to use Tensorboard on your docker container, make sure
to map the port 6006 of your docker container by adding -p 6006:6006 as shown above.

## Rebuilding the containers

Building TensorFlow Docker containers should be done through the [buildImages.sh](buildImages.sh)
script. The raw Dockerfiles should not be used directly as because the build context must be
properly prepared by the script prior to building.

A successful run of the script will:

* Copy souces into the build context
* Build both (CPU and GPU) images via `docker build`
* Export both images to `/container/images` via `docker image save`

The exported images can then be loaded into any docker registry via `docker image load`