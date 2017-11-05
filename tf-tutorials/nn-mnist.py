# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.

This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.

It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt

FLAGS = None

def train():

	def generate_embeddings():
		mnist2 = input_data.read_data_sets(FLAGS.data_dir, one_hot=False, fake_data=FLAGS.fake_data)
		images, labels = mnist2.train.next_batch(FLAGS.max_steps)
		embedding = tf.Variable(images, name="embedding")

		config = projector.ProjectorConfig()
		embed = config.embeddings.add()
		embed.tensor_name = embedding.name  # 'embedding:0'
		embed.metadata_path = 'metadata.tsv'
		embed.sprite.image_path = 'mnist_10k_sprite.png'

		# Specify the width and height of a single thumbnail.
		embed.sprite.single_image_dim.extend([28, 28])

		writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'))

		#  Generate metadata.tsv
		with open(os.path.join(FLAGS.log_dir, 'train', embed.metadata_path), 'w') as f:
			f.write("Index\tLabel\n")
			for index, label in enumerate(labels):
				f.write("%d\t%d\n" % (index, label))

		to_visualise = images
		to_visualise = vector_to_matrix_mnist(to_visualise)
		to_visualise = invert_grayscale(to_visualise)

		sprite_image = create_sprite_image(to_visualise)

		plt.imsave(os.path.join(FLAGS.log_dir, 'train', embed.sprite.image_path), sprite_image, cmap='gray')
		plt.imshow(sprite_image, cmap='gray')

		projector.visualize_embeddings(writer, config)

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

	# We can't initialize these variables to 0 - the network will get stuck.
	def weight_variable(shape):
		"""Create a weight variable with appropriate initialization."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		"""Create a bias variable with appropriate initialization."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

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

	def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
		"""Reusable code for making a simple neural net layer.

		It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
		It also sets up name scoping so that the resultant graph is easy to read,
		and adds a number of summary ops.
		"""
		# Adding a name scope ensures logical grouping of the layers in the graph.
		with tf.name_scope(layer_name):
			# This Variable will hold the state of the weights for the layer
			with tf.name_scope('weights'):
				weights = weight_variable([input_dim, output_dim])
				variable_summaries(weights)
			with tf.name_scope('biases'):
				biases = bias_variable([output_dim])
				variable_summaries(biases)
			with tf.name_scope('Wx_plus_b'):
				preactivate = tf.matmul(input_tensor, weights) + biases
				tf.summary.histogram('pre_activations', preactivate)
			activations = act(preactivate, name='activation')
			tf.summary.histogram('activations', activations)
			return activations

	def feed_dict(train):
		"""Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
		if train or FLAGS.fake_data:
			xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
			k = FLAGS.dropout
		else:
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return {x: xs, y_: ys, keep_prob: k}


	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

	sess = tf.InteractiveSession()

	# Create a multilayer model.

	# Input placeholders
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 784], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

	with tf.name_scope('input_reshape'):
		image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
		tf.summary.image('input', image_shaped_input, 10)

	hidden1 = nn_layer(x, 784, 500, 'layer1')

	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		tf.summary.scalar('dropout_keep_probability', keep_prob)
		dropped = tf.nn.dropout(hidden1, keep_prob)

	# Do not apply softmax activation yet, see below.
	y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

	with tf.name_scope('cross_entropy'):
		# The raw formulation of cross-entropy,
		#
		# tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
		#                               reduction_indices=[1]))
		#
		# can be numerically unstable.
		#
		# So here we use tf.nn.softmax_cross_entropy_with_logits on the
		# raw outputs of the nn_layer above, and then average across
		# the batch.
		diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
		with tf.name_scope('total'):
			cross_entropy = tf.reduce_mean(diff)
	tf.summary.scalar('cross_entropy', cross_entropy)

	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
			cross_entropy)

	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

	# Merge all the summaries and write them out to
	# /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
	generate_embeddings()
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()
	# saver.save(sess, os.path.join(FLAGS.log_dir, "model.ckpt"), 1)
	# Train the model, and also write summaries.
	# Every 10th step, measure test-set accuracy, and write test summaries
	# All other steps, run train_step on training data, & add training summaries

	for i in range(FLAGS.max_steps):
		if i % 10 == 0:  # Record summaries and test-set accuracy
			summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
			test_writer.add_summary(summary, i)
			print('Accuracy at step %s: %s' % (i, acc))
		else:  # Record train set summaries, and train
			if i % 100 == 99:  # Record execution stats
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				summary, _ = sess.run([merged, train_step],
									  feed_dict=feed_dict(True),
									  options=run_options,
									  run_metadata=run_metadata)
				train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
				train_writer.add_summary(summary, i)
				print('Adding run metadata for', i)

				saver.save(sess, os.path.join(FLAGS.log_dir, "model.ckpt"), i)
			else:  # Record a summary
				summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
				train_writer.add_summary(summary, i)

	saver.save(sess, os.path.join(FLAGS.log_dir, "model.ckpt"),0)

	train_writer.close()
	test_writer.close()


def main(_):
	print("Flags:", FLAGS)
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	train()


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
