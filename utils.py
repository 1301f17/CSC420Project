import numpy as np
import scipy.io as sio
import os
from PIL import Image
import random
import matplotlib.pyplot as plt

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
	fs1 = []
	xs2 = []
	ys2 = []
	fs2 = []

	turn = True

	sub_folders = [x[0] for x in os.walk(data_folder)][1:]
	random.shuffle(sub_folders)
	for sub_folder in sub_folders:
		file_prefix = sub_folder.split('/')[-1]
		image_data = rgb2gray(np.array(Image.open(sub_folder + '/' + file_prefix + '.bmp')))
		labels = sio.loadmat(sub_folder + '/' + file_prefix + "_detection.mat")['detection']

		# load patches with nucleus as center(label=1)
		for label in labels:
			if int(label[0]) > (W-1)/2 and int(label[0]) < 500 - (W-1)/2 and int(label[1]) > (H-1)/2 and int(label[1]) < 500 - (H-1)/2:
				if turn:
					xs1.append(image_data[int(label[1]-(H-1)/2):int(label[1]+(H-1)/2 + 1), int(label[0]-(W-1)/2):int(label[0]+(W-1)/2) + 1])
					ys1.append(1)
				else:
					xs2.append(image_data[int(label[1]-(H-1)/2):int(label[1]+(H-1)/2 + 1), int(label[0]-(W-1)/2):int(label[0]+(W-1)/2) + 1])
					ys2.append(1)				

		# load patches without nucleus as center(label=0)
		num_0 = 0
		for x in range(int((W-1)/2), int(500 - (W-1)/2 - 1), 5):
			for y in range(int((H-1)/2), int(500 - (H-1)/2 - 1), 5):
				flag = True
				for label in labels:
					if abs(label[0] - x) < 4 or abs(label[1] - y) < 4:
						flag = False
				if flag:
					if turn:
						xs1.append(image_data[int(y-(H-1)/2):int(y+(H-1)/2 + 1), int(x-(W-1)/2):int(x+(W-1)/2 + 1)])
						ys1.append(0)
					else:
						xs2.append(image_data[int(y-(H-1)/2):int(y+(H-1)/2 + 1), int(x-(W-1)/2):int(x+(W-1)/2 + 1)])
						ys2.append(0)
					num_0 += 1
				if num_0 > 1000:
					break

		if turn:
			fs1.append(sub_folder)
		else:
			fs2.append(sub_folder)
		turn = not turn
		print(sub_folder)
	xs1 = np.array(xs1)
	ys1 = np.array(ys1)
	fs1 = np.array(fs1)
	xs2 = np.array(xs2)
	ys2 = np.array(ys2)
	fs2 = np.array(fs2)


	np.save('xs1.npy', xs1)
	np.save('ys1.npy', ys1)
	np.save('fs1.npy', fs1)
	np.save('xs2.npy', xs2)
	np.save('ys2.npy', ys2)
	np.save('fs2.npy', fs2)


def load_data():
	xs1 = np.load('xs1.npy')
	ys1 = np.load('ys1.npy')
	fs1 = np.load('fs1.npy')
	xs2 = np.load('xs2.npy')
	ys2 = np.load('ys2.npy')
	fs2 = np.load('fs2.npy')
	return (xs1, ys1, fs1, xs2, ys2, fs2)
