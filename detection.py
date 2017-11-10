import numpy as np
import scipy.io as sio
import os
from PIL import Image
import tensorflow as tf
import random

H = 27
W = 27

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# @profile
def save_data():

	data_folder = './CRCHistoPhenotypes_2016_04_28/Detection/'
	xs1 = []
	ys1 = []
	xs2 = []
	ys2 = []

	turn = True

	sub_folders = [x[0] for x in os.walk(data_folder)][1:]
	for sub_folder in sub_folders:
		file_prefix = sub_folder.split('/')[-1]
		image_data = rgb2gray(np.array(Image.open(sub_folder + '/' + file_prefix + '.bmp')))
		labels = sio.loadmat(sub_folder + '/' + file_prefix + "_detection.mat")['detection']
		num_1 = 0
		for label in labels:
			if int(label[0]) > (W-1)/2 and int(label[0]) < 500 - (W-1)/2 and int(label[1]) > (H-1)/2 and int(label[1]) < 500 - (H-1)/2:
				if turn:
					xs1.append(image_data[int(label[1]-(H-1)/2):int(label[1]+(H-1)/2 + 1), int(label[0]-(W-1)/2):int(label[0]+(W-1)/2) + 1])
					ys1.append(1)
				else:
					xs2.append(image_data[int(label[1]-(H-1)/2):int(label[1]+(H-1)/2 + 1), int(label[0]-(W-1)/2):int(label[0]+(W-1)/2) + 1])
					ys2.append(1)					
				num_1 += 1

		num_0 = 0
		iteration = 0
		while num_0 < num_1 and iteration < 10000:
			x = random.randint(0,499)
			y = random.randint(0,499)
			flag = True
			for label in labels:
				if abs(label[0]-x) < 3 or abs(label[1]-y) < 3:
					flag = False
			if flag:
				if turn:
					xs1.append(image_data[int(y-(H-1)/2):int(y+(H-1)/2 + 1), int(x-(W-1)/2):int(x+(W-1)/2 + 1)])
					ys1.append(0)
					print(image_data[int(y-(H-1)/2):int(y+(H-1)/2 + 1), int(x-(W-1)/2):int(x+(W-1)/2 + 1)].shape)
				else:
					xs2.append(image_data[int(y-(H-1)/2):int(y+(H-1)/2 + 1), int(x-(W-1)/2):int(x+(W-1)/2 + 1)])
					ys2.append(0)					
				num_0 += 1
			iteration += 1
		turn = not turn
		print(sub_folder)
	xs1 = np.array(xs1)
	ys1 = np.array(ys1)
	xs2 = np.array(xs2)
	ys2 = np.array(ys2)
	print(xs1.shape)
	np.save('xs1.npy', xs1)
	np.save('ys1.npy', ys1)
	np.save('xs2.npy', xs2)
	np.save('ys2.npy', ys2)


def load_data():
	xs1 = np.load('xs1.npy')
	ys1 = np.load('ys1.npy')
	xs2 = np.load('xs2.npy')
	ys2 = np.load('ys2.npy')
	return (xs1, ys1, xs2, ys2)



save_data()

# (xs1, ys1, xs2, ys2) = load_data()
# print(xs1[0:2])
# print(len(ys1))
# print(sum(ys1))
# print(len(ys2))
# print(sum(ys2))


# H = 27
# W = 27

# X = tf.placeholder(tf.float32, [None, H, W], name='X')/255
# Y = tf.placeholder(tf.float32, [None, 1], name='Y')
# keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout (keep probability)


# weights = {

#     'c1': tf.Variable(tf.random_normal([4, 4, 1, 36], stddev=0.01)),

#     'c2': tf.Variable(tf.random_normal([3, 3, 36, 48], stddev=0.01)),

#     'f1': tf.Variable(tf.random_normal([5*5*48, 512], stddev=0.01)),

#     'f2': tf.Variable(tf.random_normal([512, 1], stddev=0.01)),
# }

# bias = {

#     'c1': tf.Variable(tf.constant(0, shape=[36], dtype=tf.float32)),

#     'c2': tf.Variable(tf.constant(0, shape=[48], dtype=tf.float32)),

#     'f1': tf.Variable(tf.constant(0, shape=[512], dtype=tf.float32)),

#     'f2': tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32)),

# }


# def conv2d(x, W, b, strides=1):
#     # Conv2D wrapper, with bias and relu activation
#     x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
#     x = tf.nn.bias_add(x, b)
#     return tf.nn.relu(x)


# def maxpool2d(x, k=2):
#     # MaxPool2D wrapper
#     return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
#                           padding='SAME')

# def fuc2d(x, W, b):
#     # Conv2D wrapper, with bias and relu activation
#     x = tf.nn.relu(tf.matmul(x, W) + b)
#     return tf.nn.dropout(x, keep_prob)


# X = tf.reshape(X, [-1, H, W, 1])

# c1 = conv2d(X, weights['c1'], bias['c1'])
# mp1 = maxpool2d(c1)
# c2 = conv2d(mp1, weights['c2'], bias['c2'])
# mp2 = maxpool2d(c2)
# mp2_flat = tf.reshape(mp2, [-1, 5*5*48])
# f1 = fuc2d(mp2_flat, weights['f1'], bias['f1'])
# output = tf.nn.sigmoid(tf.matmul(f1, weights['f2']) + bias['f2'])


# loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(output), reduction_indices=[1]))

# # train_step = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


# # for training and save model
# sess = tf.Session()
# init = tf.initialize_all_variables()
# sess.run(init)
# saver = tf.train.Saver()

# for i in range(120):
#     # training
#     sess.run(train_step, feed_dict={X: xs1, Y: ys1, keep_prob:0.8})
#     if i % 10 == 0:
#         # to see the step improvement
#         print(sess.run(loss, feed_dict={X: xs1, Y: ys1, keep_prob:0.8}))

# saver.save(sess, './my-model')

# # for load model and testing
# sess = tf.Session()
# saver = tf.train.import_meta_graph('./my-model.meta')
# saver.restore(sess,tf.train.latest_checkpoint('./'))

# graph = tf.get_default_graph()
# output = graph.get_tensor_by_name("output:0")
# X = graph.get_tensor_by_name("X:0")
# Y = graph.get_tensor_by_name("Y:0")
# keep_prob = graph.get_tensor_by_name("keep_prob:0")

# print(sess.run(output,feed_dict={X: xs[[0]], Y: ys[[0]], keep_prob:0.8}))

# test_image = './CRCHistoPhenotypes_2016_04_28/Detection/img93/img93.bmp'
# image_data = rgb2gray(np.array(Image.open(test_image)))

# for i in range(0, image_data.shape[0] - W):
# 	for j in range(0, image_data.shape[1] - H):
# 		patch = image_data[i:i+W, j:j+H]

