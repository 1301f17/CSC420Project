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
def save_data(color=False):

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
	# sub_folders = ['./CRCHistoPhenotypes_2016_04_28/Detection/img34', './CRCHistoPhenotypes_2016_04_28/Detection/img100', './CRCHistoPhenotypes_2016_04_28/Detection/img62', './CRCHistoPhenotypes_2016_04_28/Detection/img38', './CRCHistoPhenotypes_2016_04_28/Detection/img42', './CRCHistoPhenotypes_2016_04_28/Detection/img53', './CRCHistoPhenotypes_2016_04_28/Detection/img78', './CRCHistoPhenotypes_2016_04_28/Detection/img36', './CRCHistoPhenotypes_2016_04_28/Detection/img15', './CRCHistoPhenotypes_2016_04_28/Detection/img66', './CRCHistoPhenotypes_2016_04_28/Detection/img43', './CRCHistoPhenotypes_2016_04_28/Detection/img50', './CRCHistoPhenotypes_2016_04_28/Detection/img28', './CRCHistoPhenotypes_2016_04_28/Detection/img65', './CRCHistoPhenotypes_2016_04_28/Detection/img63', './CRCHistoPhenotypes_2016_04_28/Detection/img55', './CRCHistoPhenotypes_2016_04_28/Detection/img52', './CRCHistoPhenotypes_2016_04_28/Detection/img97', './CRCHistoPhenotypes_2016_04_28/Detection/img25', './CRCHistoPhenotypes_2016_04_28/Detection/img27', './CRCHistoPhenotypes_2016_04_28/Detection/img12', './CRCHistoPhenotypes_2016_04_28/Detection/img57', './CRCHistoPhenotypes_2016_04_28/Detection/img60', './CRCHistoPhenotypes_2016_04_28/Detection/img22', './CRCHistoPhenotypes_2016_04_28/Detection/img99', './CRCHistoPhenotypes_2016_04_28/Detection/img49', './CRCHistoPhenotypes_2016_04_28/Detection/img41', './CRCHistoPhenotypes_2016_04_28/Detection/img51', './CRCHistoPhenotypes_2016_04_28/Detection/img87', './CRCHistoPhenotypes_2016_04_28/Detection/img2', './CRCHistoPhenotypes_2016_04_28/Detection/img54', './CRCHistoPhenotypes_2016_04_28/Detection/img92', './CRCHistoPhenotypes_2016_04_28/Detection/img59', './CRCHistoPhenotypes_2016_04_28/Detection/img45', './CRCHistoPhenotypes_2016_04_28/Detection/img89', './CRCHistoPhenotypes_2016_04_28/Detection/img16', './CRCHistoPhenotypes_2016_04_28/Detection/img90', './CRCHistoPhenotypes_2016_04_28/Detection/img33', './CRCHistoPhenotypes_2016_04_28/Detection/img95', './CRCHistoPhenotypes_2016_04_28/Detection/img17', './CRCHistoPhenotypes_2016_04_28/Detection/img98', './CRCHistoPhenotypes_2016_04_28/Detection/img85', './CRCHistoPhenotypes_2016_04_28/Detection/img82', './CRCHistoPhenotypes_2016_04_28/Detection/img1', './CRCHistoPhenotypes_2016_04_28/Detection/img68', './CRCHistoPhenotypes_2016_04_28/Detection/img20', './CRCHistoPhenotypes_2016_04_28/Detection/img79', './CRCHistoPhenotypes_2016_04_28/Detection/img96', './CRCHistoPhenotypes_2016_04_28/Detection/img77', './CRCHistoPhenotypes_2016_04_28/Detection/img21', './CRCHistoPhenotypes_2016_04_28/Detection/img29', './CRCHistoPhenotypes_2016_04_28/Detection/img35', './CRCHistoPhenotypes_2016_04_28/Detection/img67', './CRCHistoPhenotypes_2016_04_28/Detection/img76', './CRCHistoPhenotypes_2016_04_28/Detection/img88', './CRCHistoPhenotypes_2016_04_28/Detection/img4', './CRCHistoPhenotypes_2016_04_28/Detection/img84', './CRCHistoPhenotypes_2016_04_28/Detection/img19', './CRCHistoPhenotypes_2016_04_28/Detection/img39', './CRCHistoPhenotypes_2016_04_28/Detection/img72', './CRCHistoPhenotypes_2016_04_28/Detection/img24', './CRCHistoPhenotypes_2016_04_28/Detection/img81', './CRCHistoPhenotypes_2016_04_28/Detection/img14', './CRCHistoPhenotypes_2016_04_28/Detection/img83', './CRCHistoPhenotypes_2016_04_28/Detection/img58', './CRCHistoPhenotypes_2016_04_28/Detection/img93', './CRCHistoPhenotypes_2016_04_28/Detection/img80', './CRCHistoPhenotypes_2016_04_28/Detection/img69', './CRCHistoPhenotypes_2016_04_28/Detection/img9', './CRCHistoPhenotypes_2016_04_28/Detection/img10', './CRCHistoPhenotypes_2016_04_28/Detection/img91', './CRCHistoPhenotypes_2016_04_28/Detection/img30', './CRCHistoPhenotypes_2016_04_28/Detection/img18', './CRCHistoPhenotypes_2016_04_28/Detection/img31', './CRCHistoPhenotypes_2016_04_28/Detection/img86', './CRCHistoPhenotypes_2016_04_28/Detection/img13', './CRCHistoPhenotypes_2016_04_28/Detection/img40', './CRCHistoPhenotypes_2016_04_28/Detection/img64', './CRCHistoPhenotypes_2016_04_28/Detection/img56', './CRCHistoPhenotypes_2016_04_28/Detection/img32', './CRCHistoPhenotypes_2016_04_28/Detection/img11', './CRCHistoPhenotypes_2016_04_28/Detection/img8', './CRCHistoPhenotypes_2016_04_28/Detection/img73', './CRCHistoPhenotypes_2016_04_28/Detection/img46', './CRCHistoPhenotypes_2016_04_28/Detection/img37', './CRCHistoPhenotypes_2016_04_28/Detection/img47', './CRCHistoPhenotypes_2016_04_28/Detection/img71', './CRCHistoPhenotypes_2016_04_28/Detection/img23', './CRCHistoPhenotypes_2016_04_28/Detection/img7', './CRCHistoPhenotypes_2016_04_28/Detection/img26', './CRCHistoPhenotypes_2016_04_28/Detection/img44', './CRCHistoPhenotypes_2016_04_28/Detection/img6', './CRCHistoPhenotypes_2016_04_28/Detection/img94', './CRCHistoPhenotypes_2016_04_28/Detection/img3', './CRCHistoPhenotypes_2016_04_28/Detection/img61', './CRCHistoPhenotypes_2016_04_28/Detection/img70', './CRCHistoPhenotypes_2016_04_28/Detection/img5', './CRCHistoPhenotypes_2016_04_28/Detection/img74', './CRCHistoPhenotypes_2016_04_28/Detection/img75', './CRCHistoPhenotypes_2016_04_28/Detection/img48']
	for sub_folder in sub_folders:
		file_prefix = sub_folder.split('/')[-1]
		image_data = rgb2gray(np.array(Image.open(sub_folder + '/' + file_prefix + '.bmp')))
		if color:
			image_data = np.array(Image.open(sub_folder + '/' + file_prefix + '.bmp'))
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

	if color:
		np.save('xs1_color.npy', xs1)
		np.save('ys1_color.npy', ys1)
		np.save('fs1_color.npy', fs1)
		np.save('xs2_color.npy', xs2)
		np.save('ys2_color.npy', ys2)
		np.save('fs2_color.npy', fs2)
	else:
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

def load_data_color():
	xs1 = np.load('xs1_color.npy')
	ys1 = np.load('ys1_color.npy')
	fs1 = np.load('fs1_color.npy')
	xs2 = np.load('xs2_color.npy')
	ys2 = np.load('ys2_color.npy')
	fs2 = np.load('fs2_color.npy')
	return (xs1, ys1, fs1, xs2, ys2, fs2)
