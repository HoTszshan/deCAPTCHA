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
import csv

file_path = "test_result_1.txt"


with open(file_path, "r") as txt_file:
    result_list = txt_file.readlines()

test_result = []
for result in result_list:
    result = result.split('\n')[0]
    split_result = result.split(' ')
    predict_time, label = split_result[2], split_result[-1]
    test_result.append([predict_time, label])

test_folder = 'annotated_captchas//test'
testing_set, testing_labels = dataset.load_captcha_dataset(test_folder)

result_length = len(test_result)
result_list = []
print result_length
for result, label in zip(test_result, testing_labels[:result_length]):
    predict_time, predict_label = result
    global_matching = 1 if predict_label == label else 0
    local_matching = sum([1 for i in range(min(len(predict_label), len(label))) if predict_label[i] == label[i]])
    result_list.append([predict_time, predict_label, label, global_matching, local_matching])
            #[str(predict_time), str(predict_label), str(label),
            #             str(global_matching), str(local_matching)])

with open('test_result.csv', 'wb') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(result_list)

