import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle
from utils import rgb2gray
from PIL import Image
import scipy.io as sio
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV



H = 27
W = 27

def train(xs1, ys1, fs1, xs2, ys2, fs2):
	xs1 = xs1.reshape(-1, H * W)
	pca = PCA(n_components=100)
	pca.fit(xs1)
	xs1 = pca.transform(xs1)
	with open('model1_pca.pkl', 'wb') as f:
		pickle.dump(pca, f)


	clf = LinearSVC()
	clf = CalibratedClassifierCV(clf)
	clf.fit(xs1, ys1)
	with open('model1_svm.pkl', 'wb') as f:
		pickle.dump(clf, f)

	xs2 = xs2.reshape(-1, H * W)
	pca = PCA(n_components=100)
	pca.fit(xs2)
	xs2 = pca.transform(xs2)
	with open('model2_pca.pkl', 'wb') as f:
		pickle.dump(pca, f)


	clf = LinearSVC()
	clf = CalibratedClassifierCV(clf)
	clf.fit(xs2, ys2)
	with open('model2_svm.pkl', 'wb') as f:
		pickle.dump(clf, f)

def cross_validation():
	TP = 0
	TP_FN = 0
	TP_FP = 0
	threshold = 0.8

	# load the model1 and test on dataset2
	with open('model1_svm.pkl', 'rb') as f:
		clf = pickle.load(f)
	with open('model1_pca.pkl', 'rb') as f:
		pca = pickle.load(f)		

	fs2 = np.load('fs2.npy')
	for sub_folder in fs2:
		print(sub_folder)
		image_path = sub_folder + '/' + sub_folder.split('/')[-1] + '.bmp'
		label_path = sub_folder + '/' + sub_folder.split('/')[-1] + '_detection.mat'
		image_data = rgb2gray(np.array(Image.open(image_path)))
		labels = sio.loadmat(label_path)['detection']

		result = np.zeros([500,500])

		# for y in range(int((H-1)/2), int(500 - (H-1)/2 - 1)):
		# 	for x in range(int((W-1)/2), int(500 - (W-1)/2 - 1)):
		# 		patch = image_data[int(y-(H-1)/2):int(y+(H-1)/2 + 1), int(x-(W-1)/2):int(x+(W-1)/2) + 1]
		# 		patch_pca = pca.transform([patch.flatten()])
		# 		result[y][x] = clf.predict(patch_pca)[0]
		patches = []
		for y in range(int((H-1)/2), int(500 - (H-1)/2 - 1)):
			# patches = []
			for x in range(int((W-1)/2), int(500 - (W-1)/2 - 1)):
				patch = image_data[int(y-(H-1)/2):int(y+(H-1)/2 + 1), int(x-(W-1)/2):int(x+(W-1)/2) + 1]
				# patch_pca = pca.transform([patch.flatten()])[0]
				patches.append(patch)
			# patches = np.array(patches)

			# print(y)
			# result[y][int((H-1)/2): int(500 - (H-1)/2 - 1)] = clf.predict(patches)
		patches = np.array(patches)
		patches = patches.reshape(-1, H*W)
		patches_pca = pca.transform(patches)
		result[int((W-1)/2): int(500 - (W-1)/2 - 1), int((H-1)/2): int(500 - (H-1)/2 - 1)] = clf.predict_proba(patches_pca)[:, 1].reshape(500-W, 500-H)

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



	# load the model2 and test on dataset1
	with open('model2_svm.pkl', 'rb') as f:
		clf = pickle.load(f)
	with open('model2_pca.pkl', 'rb') as f:
		pca = pickle.load(f)

	fs1 = np.load('fs1.npy')
	for sub_folder in fs1:
		print(sub_folder)
		image_path = sub_folder + '/' + sub_folder.split('/')[-1] + '.bmp'
		label_path = sub_folder + '/' + sub_folder.split('/')[-1] + '_detection.mat'
		image_data = rgb2gray(np.array(Image.open(image_path)))
		labels = sio.loadmat(label_path)['detection']

		result = np.zeros([500,500])

		patches = []
		for y in range(int((H-1)/2), int(500 - (H-1)/2 - 1)):
			# patches = []
			for x in range(int((W-1)/2), int(500 - (W-1)/2 - 1)):
				patch = image_data[int(y-(H-1)/2):int(y+(H-1)/2 + 1), int(x-(W-1)/2):int(x+(W-1)/2) + 1]
				# patch_pca = pca.transform([patch.flatten()])[0]
				patches.append(patch)
			# patches = np.array(patches)

			# print(y)
			# result[y][int((H-1)/2): int(500 - (H-1)/2 - 1)] = clf.predict(patches)
		patches = np.array(patches)
		patches = patches.reshape(-1, H*W)
		patches_pca = pca.transform(patches)
		result[int((W-1)/2): int(500 - (W-1)/2 - 1), int((H-1)/2): int(500 - (H-1)/2 - 1)] = clf.predict_proba(patches_pca)[:, 1].reshape(500-W, 500-H)

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


	print(TP/TP_FN)
	print(TP/TP_FP)
	print(2*TP/(TP_FN+TP_FP))	