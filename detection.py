import numpy as np
import scipy.io as sio
import os
from PIL import Image
import tensorflow as tf

H = 27
W = 27
H_ = 11
W_ = 11
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# @profile
def save_data():

	data_folder = './CRCHistoPhenotypes_2016_04_28/Detection/'
	xs = []
	ys = []
	stride = 5
	d = 6


	count = 0
	sub_folders = [x[0] for x in os.walk(data_folder)][1:]
	for sub_folder in sub_folders:
		file_prefix = sub_folder.split('/')[-1]
		image_data = rgb2gray(np.array(Image.open(sub_folder + '/' + file_prefix + '.bmp')))
		labels = sio.loadmat(sub_folder + '/' + file_prefix + "_detection.mat")['detection']
		for i in range(0, image_data.shape[0] - W, stride):
			for j in range(0, image_data.shape[1] - H, stride):
				patch = image_data[i:i+W, j:j+H]
				labels_in_omega = [label for label in labels if label[0] > i + (W-W_)/2 and label[0] < i + W - (W-W_)/2 and label[1] > j + (H-H_)/2 and label[1] < j + H - (H-H_)/2] # choose the nucleus center within Omega
				z_ms = np.array(labels_in_omega) 

				x = patch.flatten()
				if z_ms.size:
					y = []
					for p in range(W_):
						for q in range(H_):
							z_j = np.array([i + (W-W_)/2 + p, j + (H-H_)/2 + q])
							min_distance = min(np.linalg.norm(z_ms-z_j, axis=1))
							if min_distance <= d:
								y.append(1 / (min_distance**2/2 + 1))
							else:
								y.append(0)
					y = np.array(y)
				else:
					y = np.zeros(H_*W_)



				xs.append(x)
				ys.append(y)
				# if xs == None:
				# 	xs = np.array([x])
				# else:
				# 	xs = np.vstack([xs, x])

				# if ys == None:
				# 	ys = np.array([y])
				# else:
				# 	ys = np.vstack([ys, y])
		print(sub_folder)
	xs = np.array(xs)
	ys = np.array(ys)
	np.save('xs.npy', xs)
	np.save('ys.npy', ys)

def load_data():
	xs = np.load('xs.npy')
	ys = np.load('ys.npy')
	return (xs, ys)



# save_data()

(xs, ys) = load_data()	


H = 27
W = 27
H_ = 11
W_ = 11
d = 6

X = tf.placeholder(tf.float32, [None, H*W], name='X')/255
Y = tf.placeholder(tf.float32, [None, H_*W_], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout (keep probability)


weights = {

    'c1': tf.Variable(tf.random_normal([4, 4, 1, 36], stddev=0.01)),

    'c2': tf.Variable(tf.random_normal([3, 3, 36, 48], stddev=0.01)),

    'f1': tf.Variable(tf.random_normal([5*5*48, 512], stddev=0.01)),

    'f2': tf.Variable(tf.random_normal([512, 512], stddev=0.01)),

    's1_u': tf.Variable(tf.random_normal([512, 1], stddev=0.01)),
    's1_v': tf.Variable(tf.random_normal([512, 1], stddev=0.01)),
    's1_h': tf.Variable(tf.random_normal([512, 1], stddev=0.01)),
}

bias = {

    'c1': tf.Variable(tf.constant(0, shape=[36], dtype=tf.float32)),

    'c2': tf.Variable(tf.constant(0, shape=[48], dtype=tf.float32)),

    'f1': tf.Variable(tf.constant(0, shape=[512], dtype=tf.float32)),

    'f2': tf.Variable(tf.constant(0, shape=[512], dtype=tf.float32)),

    's1_u': tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32)),
    's1_v': tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32)),
    's1_h': tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32)),
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



X_image = tf.reshape(X, [-1, W, H, 1])

c1 = conv2d(X_image, weights['c1'], bias['c1'])
mp1 = maxpool2d(c1)
c2 = conv2d(mp1, weights['c2'], bias['c2'])
mp2 = maxpool2d(c2)
mp2_flat = tf.reshape(mp2, [-1, 5*5*48])
f1 = fuc2d(mp2_flat, weights['f1'], bias['f1'])
f2 = fuc2d(f1, weights['f2'], bias['f2'])
s1_u = tf.nn.sigmoid(tf.matmul(f2, weights['s1_u']) + bias['s1_u']) * tf.constant(H_-1, dtype=tf.float32) + tf.constant(1, dtype=tf.float32)
s1_v = tf.nn.sigmoid(tf.matmul(f2, weights['s1_v']) + bias['s1_v']) * tf.constant(W_-1, dtype=tf.float32) + tf.constant(1, dtype=tf.float32)
s1_h = tf.nn.sigmoid(tf.matmul(f2, weights['s1_h']) + bias['s1_h'])


# start = tf.convert_to_tensor(0.0)
# output = tf.convert_to_tensor([[0.0]])
x_coordinates = tf.convert_to_tensor(np.tile(np.arange(1, W_+1), H_), dtype=tf.float32)
y_coordinates = tf.convert_to_tensor(np.repeat(np.arange(1, H_+1), W_), dtype=tf.float32)

# x_coordinates = tf.convert_to_tensor(np.tile(np.arange(1, W_+1), (H_,1)), dtype=tf.float32)
# y_coordinates = tf.convert_to_tensor(np.tile(np.transpose([np.arange(1, H_+1)]), W_), dtype=tf.float32)


distances = ((s1_u - x_coordinates) ** 2 + (s1_v - y_coordinates) ** 2)**0.5
output = tf.select(distances < d, (1/(1 + distances**2/2))*s1_h,  distances - distances, name='output')

# for i in range(1, W_+1):
# 	for j in range(1, H_+1):
# 		# if tf.less((s1_u - j)[0] ** 2 + (s1_v - i)[0] ** 2 , d**2):
# 		# 	y = 1/((s1_u - j) ** 2 + (s1_v - i) ** 2 / 2 + 1) * s1_h
# 		# else:
# 		# 	y = tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32))
# 		# y = 1/((s1_u - j) ** 2 + (s1_v - i) ** 2 / 2 + 1) * s1_h if (s1_u - j)[0] ** 2 + (s1_v - i)[0] ** 2 < d ** 2 else tf.Variable(tf.constant(0, shape=[1], dtype=tf.float32))
# 		y = tf.select((s1_u - j) ** 2 + (s1_v - i) ** 2 < d ** 2, 1/( (s1_u - j) ** 2 / 2 + (s1_v - i) ** 2 / 2 + 1) * s1_h, s1_u - s1_u )
# 		output = tf.cond(tf.equal(start, tf.convert_to_tensor(0.0)), lambda: y, lambda: tf.concat(1, [output, y]))
# 		start = tf.convert_to_tensor(1.0)
# 		# output = tf.concat(0, [output, y])

# 		# if output == None:
# 		# 	output = y
# 		# else:
# 		# 	output = tf.concat(0, [output, y])


# epsilon = tf.count_nonzero(Y, 1, keep_dims=True, dtype=tf.float32) / (H_*W_ - tf.count_nonzero(Y, 1, keep_dims=True, dtype=tf.float32))
epsilon = 0.05
loss = tf.reduce_mean(-tf.reduce_sum((Y+epsilon)*  (Y * tf.log(tf.clip_by_value(output,1e-10,1.0)) - (1-Y)*tf.log(tf.clip_by_value(1-output,1e-10,1.0)))    ,
                                              reduction_indices=[1]))

train_step = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


# # for training and save model
# sess = tf.Session()
# init = tf.initialize_all_variables()
# sess.run(init)
# saver = tf.train.Saver()

# for i in range(120):
#     # training
#     indices = np.random.choice(xs.shape[0], 5000, replace=False)
#     sess.run(train_step, feed_dict={X: xs[indices], Y: ys[indices], keep_prob:0.8})
#     if i % 10 == 0:
#         # to see the step improvement
#         print(sess.run(loss, feed_dict={X: xs[indices], Y: ys[indices], keep_prob:0.8}))

# saver.save(sess, './my-model')

# for load model and testing
sess = tf.Session()
saver = tf.train.import_meta_graph('./my-model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
output = graph.get_tensor_by_name("output:0")
X = graph.get_tensor_by_name("X:0")
Y = graph.get_tensor_by_name("Y:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

print(sess.run(output,feed_dict={X: xs[[0]], Y: ys[[0]], keep_prob:0.8}))

test_image = './CRCHistoPhenotypes_2016_04_28/Detection/img93/img93.bmp'
image_data = rgb2gray(np.array(Image.open(test_image)))

for i in range(0, image_data.shape[0] - W):
	for j in range(0, image_data.shape[1] - H):
		patch = image_data[i:i+W, j:j+H]

