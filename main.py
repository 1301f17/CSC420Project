import utils
import detection_cnn
import detection_svm
import time

# xs1, ys1, fs1, xs2, ys2, fs2 = utils.load_data()
# detection_svm.train(xs1, ys1, fs1, xs2, ys2, fs2)

t = time.time()
detection_svm.cross_validation()
print("total runtime:")
print(time.time() - t)