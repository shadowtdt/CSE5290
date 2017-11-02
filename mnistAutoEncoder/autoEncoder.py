from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import optimizers
import tensorflow as tf


from tensorflow.python.ops import nn
import numpy as np

_LEARNING_RATE = 0.1

tf.logging.set_verbosity(tf.logging.INFO)


def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)


def ae_model_fn(features, labels, mode, params):

	hidden_units = params["hidden_units"]
	activation_fn = tf.nn.relu

	input_layer = features["x"] #tf.reshape(,[-1,784])

	# e1 = tf.layers.dense(input_layer, hidden_units*8, activation=tf.nn.relu)
	# e2 = tf.layers.dense(e1, hidden_units*4, activation=tf.nn.relu)
	# e3 = tf.layers.dense(e2, hidden_units*2, activation=tf.nn.relu)
	encoded_layer = tf.layers.dense(input_layer, hidden_units, activation=activation_fn,name="encoded_layer")
	tf.summary.tensor_summary("encoded_layer_ts", encoded_layer)
	tf.summary.histogram(encoded_layer.name+"_hist",encoded_layer)
	variable_summaries(encoded_layer)
	# d1 = tf.layers.dense(encoded_layer, hidden_units*2, activation=tf.nn.relu)
	# d2 = tf.layers.dense(d1, hidden_units*4, activation=tf.nn.relu)
	# d3 = tf.layers.dense(d2, hidden_units*8, activation=tf.nn.relu)
	decoded_layer = tf.layers.dense(encoded_layer, np.size(input_layer, axis=1), activation=activation_fn)
	def get_encoded_layer():
		return encoded_layer

	# Reshape output layer to 1-dim Tensor to return predictions
	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=encoded_layer, axis=1, name="argmax_tensor"),
	#	"labels": tf.argmax(input=labels, axis=1, name="labels"),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(encoded_layer, name="softmax_tensor"),
	}


	# Provide an estimator spec for `ModeKeys.PREDICT`.
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions)

	# Calculate loss using mean squared error
	loss = tf.losses.mean_squared_error(input_layer, decoded_layer)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdagradOptimizer(_LEARNING_RATE,)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(
			mode=mode,
			loss=loss,
			train_op=train_op)

	# Calculate root mean squared error as additional eval metric

	eval_metric_ops = {
		"rmse": tf.metrics.root_mean_squared_error(
			input_layer, decoded_layer),
	}

	# Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
	return tf.estimator.EstimatorSpec(
		mode=mode,
		loss=loss,
		eval_metric_ops=eval_metric_ops)


# Logic to do the following:
# 1. Configure the model via TensorFlow operations
# 2. Define the loss function for training/evaluation
# 3. Define the training operation/optimizer
# 4. Generate predictions
# 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object


# return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)

#
# class AutoEncoderClassifier(estimator.Estimator):
# 	def __init__(self,
# 				 hidden_units,
# 				 optimizer='Adagrad',
# 				 activation_fn="nn.relu",
# 				 config=None,
# 				 model_dir=None):
# 		def _model_fn(features, labels, mode, config):
# 			return ae_model_fn(
# 				features=features,
# 				labels=labels,
# 				mode=mode,
# 				config=config,
# 				hidden_units=hidden_units,
# 				optimizer=optimizer,
# 				activation_fn=activation_fn)
#
# 		super(AutoEncoderClassifier, self).__init__(
# 			model_fn=_model_fn, model_dir=model_dir, config=config)
