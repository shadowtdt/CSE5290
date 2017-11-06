# Unsupervised MNIST Classifier with Auto Encoder and K-Means 

This was developed in part for CSE5290 term project as the code deliverable.  

## Requirements

##### Classifier

* python
* tensorflow
* python-matplotlib

##### Visualizer (optional)

* tensorboard
* docker
* docker-compose

## Model description

There are two versions of the model, both an auto encoder and kmeans. 
The first model `autoEncoder.py` is a simple single layered auto encoder, while the second `convAutoEncoder.py` adds a convolution to the auto encoder.

The encoders are implemented using tensorflow [Estimators](https://www.tensorflow.org/versions/r1.3/programmers_guide/estimators).
While the K-Means classifier is implemented separably in `mnist_autoEncoder.py`

The models start by training a [dense](https://www.tensorflow.org/s/results/?q=dense&p=%2F) encoder layer, using [mean squared error](https://www.tensorflow.org/versions/master/api_docs/python/tf/losses/mean_squared_error) loss function optimized using [Adagrad](https://www.tensorflow.org/versions/master/api_docs/python/tf/train/AdagradOptimizer).
These models generate an encoding which is then passed to the K-Means model for clustering and classification.
K-Means model will assign a cluster id to each input. 
Clusters are then labeled based based on the most frequent digit in the cluster.
Accuracy is then computed by Correct#/Total#.
## Usage

* Run classifier:  `$ python mnist_autoEncoder --<arg>=<value> ..`
* Run visualiser: `$docker-compose up` --> `localhost:6006`

### Args
* `data_dir`: Directory to store MNIST data
* `log_dir`: Directory to store output
* `tag`: Name to give run
* `tb_embedding` (Bool): Created embedding for Tensorboard projector
* `learning_rate`: Rate at which autoencoder learns 
* `encoding_size`: The number of encodings the auto encoder will generate
* `steps`: Tumber of auto encoder training steps
* `num_clusters`: Number of K-Means clusters
* `convolution` (Bool): Use convolution when training auto encoder

## Cite
This was made possible by numerous sources and examples found on the internet and tensorflow website

*   [tomokishii/Autoencoders](https://gist.github.com/tomokishii/7ddde510edb1c4273438ba0663b26fc6)
*   [tensorflow/summaries](https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)
*   [mdda.net/Estimators](http://blog.mdda.net/ai/2017/02/25/estimator-input-fn)
*   [pkmital/autoencoder](https://github.com/pkmital/tensorflow_tutorials/blob/master/python/07_autoencoder.py)
*   [aymericdamien/kmeans](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/kmeans.py)
*   [tensorflow/Creating Estimators](https://www.tensorflow.org/extend/estimators)