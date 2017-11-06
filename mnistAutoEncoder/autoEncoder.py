import tensorflow as tf

import numpy as np

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
	hidden_units = params.get("hidden_units", 100)
	learning_rate = params.get("learning_rate", 0.05)
	activation_fn = tf.nn.relu

	input_layer = features["x"]
	# e1 = tf.layers.dense(input_layer, hidden_units*8, activation=tf.nn.relu)
	# e2 = tf.layers.dense(e1, hidden_units*4, activation=tf.nn.relu)
	# e3 = tf.layers.dense(input_layer, hidden_units*2, activation=tf.nn.relu)
	encoded_layer = tf.layers.dense(input_layer, hidden_units, activation=activation_fn, name="encoded_layer")
	# d1 = tf.layers.dense(encoded_layer, hidden_units*2, activation=tf.nn.relu)
	# d2 = tf.layers.dense(d1, hidden_units*4, activation=tf.nn.relu)
	# d3 = tf.layers.dense(d2, hidden_units*8, activation=tf.nn.relu)
	decoded_layer = tf.layers.dense(encoded_layer, np.size(input_layer, axis=1), activation=tf.nn.relu)

	tf.summary.image("image_input", tf.reshape(input_layer, [-1, 28, 28, 1]))
	#tf.summary.image("encoded_image", tf.reshape(encoded_layer, [-1, int(np.sqrt(hidden_units)), int(np.sqrt(hidden_units)), 1]))
	tf.summary.image("decoded_image", tf.reshape(decoded_layer, [-1, 28, 28, 1]))

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
		optimizer = tf.train.AdagradOptimizer(learning_rate)
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
