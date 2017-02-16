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
from multiprocessing.dummy import Pool as ThreadPool
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

"""
time_1 = time.time()
folder_dir = 'annotated_captchas//train2'
training_set, training_labels = dataset.load_captcha_dataset(folder_dir)
model = SC_KNN_Decoder(dataset='test_easy_digits', character_shape=(70, 70), sys='XOS')
time_2 = time.time()
print "Load training data: %.4f s" % (time_2 - time_1)

#model.fit(training_set, training_labels)

time_3 = time.time()
test_folder = 'annotated_captchas//test2'
testing_set, testing_labels = dataset.load_captcha_dataset(test_folder)
print "Load testing data: %.4f s" % (time.time() - time_3)


# print model.predict(testing_set)
# print testing_labels
# model.score(testing_set, testing_labels)

def save_result(index):
    lower = index
    upper = min(index+5, len(testing_labels))
    number = index / 5
    model.fast_score(testing_set[lower:upper], testing_labels[lower:upper], mode='save', paras='multi_fast'+str(number))

pool = ThreadPool(10)
pool.map(save_result, range(300, 400))
pool.close()
pool.join()
"""

