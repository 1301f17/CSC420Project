import numpy as np
import scipy.io as sio
import os
from PIL import Image
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from utils import rgb2gray

H = 27
W = 27


def train(xs1, ys1, fs1, xs2, ys2, fs2):
	X = tf.placeholder(tf.float32, [None, H, W], name='X')
	Y = tf.placeholder(tf.float32, [None, 1], name='Y')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout (keep probability)


	weights = {

	    'c1': tf.Variable(tf.random_normal([4, 4, 1, 36], stddev=0.01)),

	    'c2': tf.Variable(tf.random_normal([3, 3, 36, 48], stddev=0.01)),

	    'f1': tf.Variable(tf.random_normal([5*5*48, 512], stddev=0.01)),

	    'f2': tf.Variable(tf.random_normal([512, 1], stddev=0.01)),
	}

	bias = {

	    'c1': tf.Variable(tf.constant(0, shape=[36], dtype=tf.float32)),

	    'c2': tf.Variable(tf.constant(0, shape=[48], dtype=tf.float32)),

	    'f1': tf.Variable(tf.constant(0, shape=[512], dtype=tf.float32)),

	    'f2': tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32)),

	}


	def conv2d(x, W, b, strides=1):
	    # Conv2D wrapper, with bias and relu activation
	    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
	    x = tf.nn.bias_add(x, b)
	    return tf.nn.relu(x)


	def maxpool2d(x, k=2):
	    # MaxPool2D wrapper
	    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
	                          padding='SAME')

	def fuc2d(x, W, b):
	    # Conv2D wrapper, with bias and relu activation
	    x = tf.nn.relu(tf.matmul(x, W) + b)
	    return tf.nn.dropout(x, keep_prob)


	X = tf.reshape(X, [-1, H, W, 1])

	c1 = conv2d(X, weights['c1'], bias['c1'])
	mp1 = maxpool2d(c1)
	c2 = conv2d(mp1, weights['c2'], bias['c2'])
	mp2 = maxpool2d(c2)
	mp2_flat = tf.reshape(mp2, [-1, 5*5*48])
	f1 = fuc2d(mp2_flat, weights['f1'], bias['f1'])
	output = tf.nn.sigmoid(tf.matmul(f1, weights['f2']) + bias['f2'], name='output')
	# print(c1.get_shape())
	# print(mp1.get_shape())
	# print(c2.get_shape())
	# print(mp2.get_shape())
	# print(mp2_flat.get_shape())
	# print(f1.get_shape())
	# print(output.get_shape())

	loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(output) + (1-Y) * tf.log(1 - output), reduction_indices=[1]))

	# train_step = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(loss)
	train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


	# train and save model1
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	saver = tf.train.Saver()

	for i in range(300):
	    # training
	    indices = np.random.choice(len(xs1), 5000, replace=False)
	    sess.run(train_step, feed_dict={X: xs1[indices].reshape(-1, H, W, 1), Y: ys1[indices].reshape(-1, 1), keep_prob:0.8})
	    if i % 10 == 0:
	        # to see the step improvement
	        print(sess.run(loss, feed_dict={X: xs1[indices].reshape(-1, H, W, 1), Y: ys1[indices].reshape(-1, 1), keep_prob:0.8}))
	        # print(sess.run(output, feed_dict={X: xs1[indices].reshape(-1, H, W, 1), keep_prob:0.8}))


	saver.save(sess, './model1/model1')
	sess.close()

	# train and save model1
	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	saver = tf.train.Saver()

	for i in range(300):
	    # training
	    indices = np.random.choice(len(xs2), 5000, replace=False)
	    sess.run(train_step, feed_dict={X: xs2[indices].reshape(-1, H, W, 1), Y: ys2[indices].reshape(-1, 1), keep_prob:0.8})
	    if i % 10 == 0:
	        # to see the step improvement
	        print(sess.run(loss, feed_dict={X: xs2[indices].reshape(-1, H, W, 1), Y: ys2[indices].reshape(-1, 1), keep_prob:0.8}))

	saver.save(sess, './model2/model2')
	sess.close()

def cross_validation():
	TP = 0
	TP_FN = 0
	TP_FP = 0
	threshold = 0.98

	# load the model1 and test on dataset2
	sess = tf.Session()
	saver = tf.train.import_meta_graph('./model1/model1.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./model1/'))

	graph = tf.get_default_graph()
	output = graph.get_tensor_by_name("output:0")
	X = graph.get_tensor_by_name("X:0")
	Y = graph.get_tensor_by_name("Y:0")
	keep_prob = graph.get_tensor_by_name("keep_prob:0")

	fs2 = np.load('fs2.npy')
	for sub_folder in fs2:
		print(sub_folder)
		image_path = sub_folder + '/' + sub_folder.split('/')[-1] + '.bmp'
		label_path = sub_folder + '/' + sub_folder.split('/')[-1] + '_detection.mat'
		image_data = rgb2gray(np.array(Image.open(image_path)))
		labels = sio.loadmat(label_path)['detection']

		result = np.zeros([500,500])

		for y in range(int((H-1)/2), int(500 - (H-1)/2 - 1)):
			patches = []
			for x in range(int((W-1)/2), int(500 - (W-1)/2 - 1)):
				patch = image_data[int(y-(H-1)/2):int(y+(H-1)/2 + 1), int(x-(W-1)/2):int(x+(W-1)/2) + 1]
				patches.append(patch)
			patches = np.array(patches)
			result[y][int((H-1)/2): int(500 - (H-1)/2 - 1)] = sess.run(output,feed_dict={X: patches, keep_prob:0.8}).reshape(1, -1)

		# filter the result so that only those local max and greater than some threshold will be positive.
		result_filtered = np.zeros([500,500])
		for y in range(6, result.shape[0]-7):
			for x in range(6, result.shape[1]-7):
				if result[y,x] == np.amax(result[y-6:y+7,x-6:x+7]) and result[y,x] > threshold:
					result_filtered[y,x] = 1

		# calculate TP FP FN for precision-recall
		for label in labels:
			if int(label[0]) > 6 and int(label[0]) < 493 and int(label[1]) > 6 and int(label[1]) < 393:
				TP_FN += 1
				if sum(sum(result_filtered[int(label[1])-6:int(label[1])+7, int(label[0])-6:int(label[0])+7])) > 0: # there exists at least one positive at the region of radius=6
					TP += 1
		TP_FP += sum(sum(result_filtered))

	sess.close()
	tf.reset_default_graph()


	# load the model2 and test on dataset1
	sess = tf.Session()
	saver = tf.train.import_meta_graph('./model2/model2.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./model2/'))

	graph = tf.get_default_graph()
	output = graph.get_tensor_by_name("output:0")
	X = graph.get_tensor_by_name("X:0")
	Y = graph.get_tensor_by_name("Y:0")
	keep_prob = graph.get_tensor_by_name("keep_prob:0")


	fs1 = np.load('fs1.npy')
	for sub_folder in fs1:
		print(sub_folder)
		image_path = sub_folder + '/' + sub_folder.split('/')[-1] + '.bmp'
		label_path = sub_folder + '/' + sub_folder.split('/')[-1] + '_detection.mat'
		image_data = rgb2gray(np.array(Image.open(image_path)))
		labels = sio.loadmat(label_path)['detection']

		result = np.zeros([500,500])

		for y in range(int((H-1)/2), int(500 - (H-1)/2 - 1)):
			patches = []
			for x in range(int((W-1)/2), int(500 - (W-1)/2 - 1)):
				patch = image_data[int(y-(H-1)/2):int(y+(H-1)/2 + 1), int(x-(W-1)/2):int(x+(W-1)/2) + 1]
				patches.append(patch)
			patches = np.array(patches)
			result[y][int((H-1)/2): int(500 - (H-1)/2 - 1)] = sess.run(output,feed_dict={X: patches, keep_prob:0.8}).reshape(1, -1)

		# filter the result so that only those local max and greater than some threshold will be positive.
		result_filtered = np.zeros([500,500])
		for y in range(6, result.shape[0]-7):
			for x in range(6, result.shape[1]-7):
				if result[y,x] == np.amax(result[y-6:y+7,x-6:x+7]) and result[y,x] > threshold:
					result_filtered[y,x] = 1

		# calculate TP FP FN for precision-recall
		for label in labels:
			if int(label[0]) > 6 and int(label[0]) < 493 and int(label[1]) > 6 and int(label[1]) < 493:
				TP_FN += 1
				if sum(sum(result_filtered[int(label[1])-6:int(label[1])+7, int(label[0])-6:int(label[0])+7])) > 0: # there exists at least one positive at the region of radius=6
					TP += 1
		TP_FP += sum(sum(result_filtered))

	sess.close()

	print(TP/TP_FN)
	print(TP/TP_FP)
	print(2*TP/(TP_FN+TP_FP))



