import tensorflow as tf

import numpy as np

_LEARNING_RATE = 0.5
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


def conv_ae_model_fn(features, labels, mode, params):
	hidden_units = params["hidden_units"]
	activation_fn = tf.nn.relu

	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
	# Encode
	enConv1 = tf.layers.conv2d(input_layer, 10, 3,activation=activation_fn)
	enPool1 = tf.layers.max_pooling2d(enConv1, pool_size=3, strides=1, padding="SAME")
	enConv2 = tf.layers.conv2d(enPool1, 10, 3,activation=activation_fn)
	enPool2 = tf.layers.max_pooling2d(enConv2, pool_size=3, strides=1, padding="SAME")
	enConv3 = tf.layers.conv2d(enPool2, 10, 3,activation=activation_fn)
	enPool3 = tf.layers.max_pooling2d(enConv3, pool_size=3, strides=1, padding="SAME")

	encoded_layer = tf.layers.dense(enPool3, hidden_units, activation=activation_fn, name="encoded_layer")
	# Decode
	deTConv1 = tf.layers.conv2d_transpose(encoded_layer, 10, 3,activation=activation_fn)
	deTConv2 = tf.layers.conv2d_transpose(deTConv1, 10, 3,activation=activation_fn)
	deTConv3 = tf.layers.conv2d_transpose(deTConv2, 10, 3,activation=activation_fn)

	decoded_layer = tf.layers.conv2d(deTConv3,1,1,activation=tf.nn.sigmoid)

	# decoded_layer = tf.reshape(deTConv3, [-1, 28, 28, 1])

	tf.summary.image("image_input", input_layer)
	# tf.summary.image("encoded_image", tf.reshape(encoded_layer, [-1, int(np.sqrt(hidden_units)), int(np.sqrt(hidden_units)), 1]))
	tf.summary.image("decoded_image", decoded_layer)

	# Reshape output layer to 1-dim Tensor to return predictions
	predictions = {
		"input": input_layer,
		"encoded": encoded_layer,
		"decoded": decoded_layer
	}

	with tf.name_scope("predictions"):
		for k, v in predictions.iteritems():
			if v is not None:
				with tf.name_scope(k):
					variable_summaries(v)

	# Provide an estimator spec for `ModeKeys.PREDICT`.
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions)

	# Calculate loss using mean squared error
	with tf.name_scope("loss"):
		loss = tf.losses.mean_squared_error(input_layer, decoded_layer)
		variable_summaries(loss)

	# Provide an estimator spec for `ModeKeys.TRAIN`.
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdagradOptimizer(_LEARNING_RATE)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(
			mode=mode,
			loss=loss,
			train_op=train_op)

	# Calculate root mean squared error as additional eval metric
	eval_metric_ops = {
		"rmse": tf.metrics.root_mean_squared_error(input_layer, decoded_layer),
	}

	# Provide an estimator spec for `ModeKeys.EVAL` modes.
	return tf.estimator.EstimatorSpec(
		mode=mode,
		loss=loss,
		eval_metric_ops=eval_metric_ops)
