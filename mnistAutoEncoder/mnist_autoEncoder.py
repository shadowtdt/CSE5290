import argparse
import os
import sys

from autoEncoder import ae_model_fn
from convAutoEncoder import conv_ae_model_fn

import numpy as np
import csv

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import factorization as fact, learn as ln

import scipy.misc
def run():

	def add_embeddings(config, image_data, label_dict, embedded, name):
		embedding = tf.Variable(np.array(embedded), name=name)

		embed = config.embeddings.add()
		embed.tensor_name = embedding.name  # 'embedding:0'
		embed.metadata_path = name + '_metadata.tsv'
		embed.sprite.image_path = name + '_image_sprite.png'

		# Specify the width and height of a single thumbnail.
		embed.sprite.single_image_dim.extend([28, 28])

		#  Generate metadata.tsv
		with open(os.path.join(tag_dir, embed.metadata_path), 'w') as f:
			writer = csv.writer(f, delimiter="\t")
			# Add column headers if there are multiple
			if len(label_dict.keys()) > 1:
				writer.writerow(label_dict.keys())
			writer.writerows(zip(*label_dict.values()))

		to_visualise = image_data
		to_visualise = vector_to_matrix_mnist(to_visualise)
		to_visualise = invert_grayscale(to_visualise)

		sprite_image = create_sprite_image(to_visualise)

		scipy.misc.toimage(sprite_image, cmin=0.0).save(os.path.join(tag_dir, embed.sprite.image_path))


	def create_sprite_image(image_data):
		"""Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
		if isinstance(image_data, list):
			image_data = np.array(image_data)
		img_h = image_data.shape[1]
		img_w = image_data.shape[2]
		n_plots = int(np.ceil(np.sqrt(image_data.shape[0])))

		spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

		for i in range(n_plots):
			for j in range(n_plots):
				this_filter = i * n_plots + j
				if this_filter < image_data.shape[0]:
					this_img = image_data[this_filter]
					spriteimage[i * img_h:(i + 1) * img_h,
					j * img_w:(j + 1) * img_w] = this_img

		return spriteimage

	def vector_to_matrix_mnist(mnist_digits):
		"""Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
		return np.reshape(mnist_digits, (-1, 28, 28))

	def invert_grayscale(mnist_digits):
		""" Makes black white, and white black """
		return 1 - mnist_digits

	# Setup
	tag_dir = os.path.join(FLAGS.log_dir, FLAGS.tag)
	model = ae_model_fn
	if FLAGS.convolution:
		model = conv_ae_model_fn
	summary_writer = tf.summary.FileWriter(tag_dir)
	projector_config = projector.ProjectorConfig()

	# MNIST DATA
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

	sess = tf.InteractiveSession()

	## Auto Encoder ##

	# Auto Encoder Model setup
	ae_params = {
		"hidden_units": FLAGS.encoding_size,
		"learning_rate": FLAGS.learning_rate
	}
	run_config = ln.RunConfig(save_summary_steps=100, model_dir=tag_dir)
	encoder_model = tf.estimator.Estimator(model_fn=model, model_dir=tag_dir, params=ae_params, config=run_config)

	# TRAIN
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(mnist.train.images)},
		y=np.array(mnist.train.labels),
		num_epochs=None,
		shuffle=True)
	encoder_model.train(input_fn=train_input_fn, steps=FLAGS.steps, hooks=[])

	# TEST
	test_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(mnist.test.images)},
		y=np.array(mnist.test.labels),
		shuffle=True)
	eval = encoder_model.evaluate(input_fn=test_input_fn)
	print("Loss: %s" % eval["loss"])
	print("Root Mean Squared Error: %s" % eval["rmse"])

	# PREDICT ~ Generate encoding
	images = mnist.test.images
	labels = mnist.test.labels
	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(images)},
		y=np.array(labels),
		shuffle=False)
	model_results = encoder_model.predict(input_fn=predict_input_fn)

	encoding_train = []
	for p in model_results:
		encoding_train.append(p["encoded"].flatten())


	## K-MEANS ##
	kmean_input = encoding_train

	# Parameters
	num_steps = 30  # Total steps to train
	num_clusters = FLAGS.num_clusters  # The number of clusters
	num_classes = 10  # The 10 digits
	num_features = len(encoding_train[0]) # Features based on encoder output
	print("Number of features: %d" % num_features)
	print("Number of clusters: %d" % num_clusters)

	# Setup
	X = tf.placeholder(tf.float32, shape=[None, num_features])
	Y = tf.placeholder(tf.float32, shape=[None, num_classes])

	# K-Means Model setup
	k_means = fact.KMeans(inputs=X, num_clusters=num_clusters, distance_metric='cosine', use_mini_batch=True)
	(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op, train_op) = k_means.training_graph()
	cluster_idx = cluster_idx[0]
	avg_distance = tf.reduce_mean(scores)

	# Init model
	init_vars = tf.global_variables_initializer()
	sess.run(init_vars, feed_dict={X: kmean_input})
	sess.run(init_op, feed_dict={X: kmean_input})

	# TRAIN
	for i in range(1, num_steps + 1):
		_, d, idx = sess.run([train_op, avg_distance, cluster_idx], feed_dict={X: kmean_input})
		if i % 10 == 0 or i == 1:
			print("Step %i, Avg Distance: %f" % (i, d))
		# print(Counter(idx))

	# TEST

	# Label each cluster based on most common member label
	labels = np.argmax(labels, axis=1)
	counts = np.zeros(shape=(num_clusters, num_classes))
	for i in range(len(idx)):
		counts[idx[i]][labels[i]] += 1
	labels_map = tf.convert_to_tensor([np.argmax(c) for c in counts])

	# Cluster mapping (cluster id -> digit label)
	cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)

	# Compute accuracy
	correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
	accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	images_test = mnist.train.images
	labels_test = mnist.train.labels
	test_input_fun = tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(images_test)},
		y=np.array(labels_test),
		shuffle=False)
	model_result = encoder_model.predict(input_fn=test_input_fun)

	encoding_test = []
	for p in model_result:
		encoding_test.append(p["encoded"].flatten())

	# Classify with K-Means using auto-encoded features
	accuracy, idx_test, prediction_success = sess.run([accuracy_op, cluster_idx, correct_prediction], feed_dict={X: encoding_test, Y: labels_test})
	print("Test Accuracy: %f" % accuracy)

	## SAVE

	# Create embeddings for tensorboard
	if FLAGS.tb_embedding:
		labels_test = np.argmax(labels_test, axis=1)
		add_embeddings(projector_config, images, {"clusterID": idx, "label": labels}, encoding_train, "encoded_embedding")
		add_embeddings(projector_config, images_test, {"clusterID": idx_test, "label": labels_test, "correct_prediction": prediction_success}, encoding_test, "encoded_embedding_test")

	# Save model, events, embeddings
	projector.visualize_embeddings(summary_writer, projector_config)
	saver = tf.train.Saver()
	saver.save(sess, os.path.join(tag_dir, "model.ckpt"), 0)


def main(self):
	print("TF Version  (Tested with 1.30): ", tf.VERSION)
	print("Flags:", FLAGS)

	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)

	run()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument('--max_steps', type=int, default=1000,help='Number of steps to run trainer.')
	# parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False, help='If true, uses fake data for unit testing.')
	parser.add_argument('--data_dir', type=str, default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/input_data'), help='Directory for storing input data')
	parser.add_argument('--log_dir', type=str, default="summary", help='Summaries log directory')
	parser.add_argument('--tag', type=str, default="run0", help='Tag to save results under')
	parser.add_argument('--tb_embedding', type=bool, default=True, help='If true, generates the projector embeddings for tensorboard')
	parser.add_argument('--learning_rate', type=float, default=0.04, help='Initial learning rate')
	parser.add_argument('--encoding_size', type=int, default=144, help='The number of latent features to encode')
	parser.add_argument('--steps', type=int, default=5000, help='Number of Auto Encoder training steps')
	parser.add_argument('--num_clusters', type=int, default=360, help='Number of K-Mean clusters')
	parser.add_argument('--convolution', type=bool, default=False, help='Number of K-Mean clusters')

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
