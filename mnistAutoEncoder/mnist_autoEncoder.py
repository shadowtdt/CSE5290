from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from autoEncoder import ae_model_fn

import numpy as np
import csv

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import factorization as fact, learn as ln
import matplotlib.pyplot as plt

_HIDDEN_UNITS = int(np.square(12))

_RUN_DIR = "logs"
_SUMMARY_WRITER = None
_CONFIG = projector.ProjectorConfig()


def run():
	def add_embeddings(config, images, label_dict, embedded, name):
		embedding = tf.Variable(np.array(embedded), name=name)

		embed = config.embeddings.add()
		embed.tensor_name = embedding.name  # 'embedding:0'
		embed.metadata_path = name + '_metadata.tsv'
		embed.sprite.image_path = name + '_image_sprite.png'

		# Specify the width and height of a single thumbnail.
		embed.sprite.single_image_dim.extend([28, 28])

		#  Generate metadata.tsv
		with open(os.path.join(_RUN_DIR, embed.metadata_path), 'w') as f:
			writer = csv.writer(f, delimiter="\t")
			if len(label_dict.keys()) > 1:
				writer.writerow(label_dict.keys())
			writer.writerows(zip(*label_dict.values()))

		# #  Generate metadata.tsv
		# with open(os.path.join(_RUN_DIR,embed.metadata_path), 'w') as f:
		# 	f.write("Index\tLabel\n")
		# 	for index, label in enumerate(labels):
		# 		f.write("%d\t%d\n" % (index, label))

		to_visualise = images
		to_visualise = vector_to_matrix_mnist(to_visualise)
		to_visualise = invert_grayscale(to_visualise)

		sprite_image = create_sprite_image(to_visualise)

		plt.imsave(os.path.join(_RUN_DIR, embed.sprite.image_path), sprite_image, cmap='gray')
		plt.imshow(sprite_image, cmap='gray')

	def create_sprite_image(images):
		"""Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
		if isinstance(images, list):
			images = np.array(images)
		img_h = images.shape[1]
		img_w = images.shape[2]
		n_plots = int(np.ceil(np.sqrt(images.shape[0])))

		spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

		for i in range(n_plots):
			for j in range(n_plots):
				this_filter = i * n_plots + j
				if this_filter < images.shape[0]:
					this_img = images[this_filter]
					spriteimage[i * img_h:(i + 1) * img_h,
					j * img_w:(j + 1) * img_w] = this_img

		return spriteimage

	def vector_to_matrix_mnist(mnist_digits):
		"""Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
		return np.reshape(mnist_digits, (-1, 28, 28))

	def invert_grayscale(mnist_digits):
		""" Makes black white, and white black """
		return 1 - mnist_digits

	print("TF Version: ", tf.VERSION)
	_RUN_DIR = os.path.join(FLAGS.log_dir, "run0")
	_SUMMARY_WRITER = tf.summary.FileWriter(_RUN_DIR)

	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

	sess = tf.Session()

	run_config = ln.RunConfig(save_summary_steps=100, model_dir=_RUN_DIR)
	ae_params = {"hidden_units": _HIDDEN_UNITS}
	autoE = tf.estimator.Estimator(model_fn=ae_model_fn, model_dir=_RUN_DIR, params=ae_params, config=run_config)

	# TRAIN
	# Set up logging for predictions
	# tensors_to_log = {
	# 	# "labels": "labels",
	# 	"probabilities": "softmax_tensor"}
	# logging_hook = tf.train.LoggingTensorHook(
	# 	tensors=tensors_to_log, every_n_iter=1000)

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(mnist.train.images)},
		y=np.array(mnist.train.labels),
		num_epochs=None,
		shuffle=True)
	autoE.train(input_fn=train_input_fn, steps=30000, hooks=[])

	# TEST
	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(mnist.test.images)},
		y=np.array(mnist.test.labels),  # np.array(mnist.train.labels),
		shuffle=True)
	eval = autoE.evaluate(input_fn=test_input_fn)
	print("Loss: %s" % eval["loss"])
	print("Root Mean Squared Error: %s" % eval["rmse"])

	# PREDICT
	images = mnist.train.images
	labels = mnist.train.labels
	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(images)},
		y=np.array(labels),
		shuffle=False)
	predictGen = autoE.predict(input_fn=predict_input_fn)

	probs = []
	for p in predictGen:
		probs.append(list(p["probabilities"]))

	# K-MEANS
	full_data_x = probs

	# Parameters
	num_steps = 30  # Total steps to train
	k = 1000  # The number of clusters
	num_classes = 10  # The 10 digits
	num_features = _HIDDEN_UNITS

	# Input images
	X = tf.placeholder(tf.float32, shape=[None, num_features])
	# Labels (for assigning a label to a centroid and testing)
	Y = tf.placeholder(tf.float32, shape=[None, num_classes])

	# K-Means Parameters
	k_means = fact.KMeans(inputs=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)

	# Build KMeans graph
	(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, train_op) = k_means.training_graph()
	cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
	avg_distance = tf.reduce_mean(scores)

	# Initialize the variables (i.e. assign their default value)
	init_vars = tf.global_variables_initializer()

	# Start TensorFlow session
	# sess = tf.Session()

	# Run the initializer
	sess.run(init_vars, feed_dict={X: full_data_x})
	sess.run(init_op, feed_dict={X: full_data_x})

	# Training
	for i in range(1, num_steps + 1):
		_, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={X: full_data_x})
		if i % 10 == 0 or i == 1:
			print("Step %i, Avg Distance: %f" % (i, d))

	# Assign a label to each centroid
	# Count total number of labels per centroid, using the label of each training
	# sample to their closest centroid (given by 'idx')
	labels = np.argmax(labels, axis=1)
	counts = np.zeros(shape=(k, num_classes))
	for i in range(len(idx)):
		counts[idx[i]][labels[i]] += 1

	# Assign the most frequent label to the centroid
	labels_map = tf.convert_to_tensor([np.argmax(c) for c in counts])

	# Evaluation ops
	# Lookup: centroid_id -> label
	cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
	# Compute accuracy
	correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
	accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Test Model
	# test_x, test_y = mnist.test.images, mnist.test.labels

	images_test = mnist.test.images
	labels_test = mnist.test.labels
	predict_input_fn2 = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(images_test)},
		y=np.array(labels_test),
		num_epochs=1,
		shuffle=False)
	predictGen2 = autoE.predict(input_fn=predict_input_fn2)

	probs_test = []

	for p in predictGen2:
		probs_test.append(list(p["probabilities"]))

	accuracy, idx_test, prediction_success = sess.run([accuracy_op, cluster_idx, correct_prediction],
													  feed_dict={X: probs_test, Y: labels_test})
	print("Test Accuracy:", accuracy)

	labels_test = np.argmax(labels_test, axis=1)
	add_embeddings(_CONFIG, images, {"clusterID": idx, "label": labels}, probs, "encoded_embedding")
	add_embeddings(_CONFIG, images_test,
				   {"clusterID": idx_test, "label": labels_test, "correct_prediction": prediction_success}, probs_test,
				   "encoded_embedding_test")

	sess.run(tf.global_variables_initializer())
	projector.visualize_embeddings(_SUMMARY_WRITER, _CONFIG)
	saver = tf.train.Saver()
	saver.save(sess, os.path.join(_RUN_DIR, "model.ckpt"), 0)


def main(_):
	print("Flags:", FLAGS)
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
						default=False,
						help='If true, uses fake data for unit testing.')
	parser.add_argument('--max_steps', type=int, default=1000,
						help='Number of steps to run trainer.')
	parser.add_argument('--learning_rate', type=float, default=0.001,
						help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.9,
						help='Keep probability for training dropout.')
	parser.add_argument(
		'--data_dir',
		type=str,
		default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
							 'tensorflow/mnist/input_data'),
		help='Directory for storing input data')
	parser.add_argument(
		'--log_dir',
		type=str,
		default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
							 'tensorflow/mnist/logs/mnist_with_summaries'),
		help='Summaries log directory')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
