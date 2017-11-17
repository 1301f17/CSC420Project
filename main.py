import utils
import detection_cnn
import detection_svm
import detection_svm_color
import time

# utils.save_data()

xs1, ys1, fs1, xs2, ys2, fs2 = utils.load_data()

detection_cnn.train(xs1, ys1, fs1, xs2, ys2, fs2)
detection_cnn.cross_validation()

detection_svm.train(xs1, ys1, fs1, xs2, ys2, fs2)
detection_svm.cross_validation()


xs1, ys1, fs1, xs2, ys2, fs2 = utils.load_data_color()
detection_svm_color.train(xs1, ys1, fs1, xs2, ys2, fs2)
detection_svm_color.cross_validation()
