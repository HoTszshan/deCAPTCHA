from lib import imgio
from lib import dataset
from lib import process
from lib.segment import CharacterSeparator
from feature.shapecontext import *
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform


import os
import random
import math
import numpy as np
import numpy.linalg as nl

import time
import pickle


from PIL import Image
from scipy import sparse
from skimage import feature as ski_feature


from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

from sc_knn_decoder import *
from feature import shapecontext_1
import sys
sys.setrecursionlimit(10000)

"""
file_path = 'annotated_captchas\\train\\9291-0.jpg'
image = imgio.read_img_uc(file_path)
image = pre_processing_digit(image)
#imgio.show_img(image)

separator = CharacterSeparator(image, character_shape=(100, 100))
separator.segment_process()
separator.show_split_chunks()
separator.show_split_objects()
#separator.save_segment_result('tmp', '0082')
#"""

time_1 = time.time()

folder_dir = 'annotated_captchas//train'
training_set, training_labels = dataset.load_captcha_dataset(folder_dir)

model = SC_KNN_Decoder(dataset='easy_digits', character_shape=(70, 70), sys='XOS')

time_2 = time.time()
print "Load label: %.4f s" % (time_2 - time_1)

model.fit(training_set, training_labels)

time_3 = time.time()
test_folder = 'annotated_captchas//test'
testing_set, testing_labels = dataset.load_captcha_dataset(test_folder)
print "Load test data:", time.time() - time_3


# print model.predict(testing_set)
# print testing_labels
model.score(testing_set, testing_labels)
