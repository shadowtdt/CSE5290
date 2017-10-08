tf-tutorials is a project that implements some of the tutorials found on the TesnsorFlow website
* [Getting started]
* [Tutorials]

[Getting started]:https://www.tensorflow.org/get_started/
[Tutorials]:https://www.tensorflow.org/tutorials/

Project contains shared [PyCharm][][run configurations][] for running the tutorials.
You will have to update the python [interpreter][] for you local system, I suggest using a [python virtual environment][], or [Docker][] 

[pycharm]:https://www.jetbrains.com/pycharm/
[run configurations]:https://www.jetbrains.com/help/pycharm/creating-and-editing-run-debug-configurations.html
[interpreter]:https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html
[python virtual environment]:https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html#configuring-venv
[Docker]:https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-docker-compose.html

# Usage

## nn-mnist
Running `nn-mnist.py` will train a simple neural network on the MNIST dataset, and produce TensorFlow [summaries][] and [checkpoints][].
The test set accuracy is displayed on the stdout/terminal.
Additionally, you can visualize the results with [TensorBoard][] by running the `docker-compose.yml`, and connecting to [localhost:6006][]. 

[summaries]:https://www.tensorflow.org/get_started/summaries_and_tensorboard
[checkpoints]:https://www.tensorflow.org/programmers_guide/saved_model
[TensorBoard]:https://www.tensorflow.org/get_started/summaries_and_tensorboard
[localhost:6006]:http://localhost:6006
