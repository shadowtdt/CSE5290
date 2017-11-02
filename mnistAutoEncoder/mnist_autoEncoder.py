from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
from autoEncoder import ae_model_fn

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt


def run():

	# a = [1,1,2,2,3,3]
	# b = ["a","b","c","a","b","c"]
	#
	#
	# ziped = zip(a,b)
	# dicZip = dict(ziped)
	#
	# print(ziped)
	# print(dicZip)
	# d = {}
	# for k, v in ziped:
	# 	if d.get(k) == None:
	# 		d.__setitem__(k, [])
	# 	d[k].append(v)
	#
	# print(d)

	def generate_embeddings(images, labels, embeded_prediction):
		#mnist2 = input_data.read_data_sets(FLAGS.data_dir, one_hot=False, fake_data=FLAGS.fake_data)
		#images, labels = mnist2.train.next_batch(FLAGS.max_steps)
		embedding = tf.Variable(np.array(embeded_prediction), name="embedding")

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

	print("Version: ", tf.VERSION)
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

	sess = tf.InteractiveSession()

	ae_params = {"hidden_units": 100}
	autoE = tf.estimator.Estimator(model_fn=ae_model_fn,model_dir=FLAGS.log_dir, params=ae_params)

	# TRAIN
	# Set up logging for predictions
	tensors_to_log = {
		# "labels": "labels",
		"classes": "argmax_tensor",
		"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=1000)
	# Set up summaries
	# train_summary_hook = tf.train.SummarySaverHook(
	# 	save_steps=100,
	# 	output_dir=os.path.join(FLAGS.log_dir, "train"))
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(mnist.train.images)},
		y=np.array(mnist.train.labels),  # np.array(mnist.train.labels),
		batch_size=1000,
		num_epochs=100,
		shuffle=True)
	autoE.train(input_fn=train_input_fn, steps=10000, hooks=[logging_hook])

	# TEST
	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(mnist.test.images)},
		y=np.array(mnist.test.labels),  # np.array(mnist.train.labels),
		num_epochs=1,
		shuffle=True)
	eval = autoE.evaluate(input_fn=test_input_fn)
	print("Loss: %s" % eval["loss"])
	print("Root Mean Squared Error: %s" % eval["rmse"])

	# PREDICT
	images = mnist.train.images
	labels = mnist.train.labels
	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(images)},
		y=np.array(labels),  # np.array(mnist.train.labels),
		num_epochs=1,
		shuffle=False)
	predictGen = autoE.predict(input_fn=predict_input_fn)

	probs = []
	classes = []

	for p in predictGen:
		probs.append(list(p["probabilities"]))
		classes.append(p["classes"])

	for predictions in predictGen:
		classes.append(predictions["classes"])

	labels = np.argmax(labels,axis=1)
	generate_embeddings(images,labels,probs)

	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.save(sess, os.path.join(FLAGS.log_dir, "model.ckpt"),0)


	# print(classes)
	# pairs = zip(labels,classes)
	# labelMap = {}
	# for k, v in pairs:
	# 	if labelMap.get(k) == None:
	# 		labelMap.__setitem__(k, [None])
	# 	labelMap[k].append(v)
	#
	# for item in labelMap:
	# 	print(item)
	# 	print(labelMap[item])

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
