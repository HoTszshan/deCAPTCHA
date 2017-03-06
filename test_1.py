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
import sys
sys.setrecursionlimit(10000)

# basic operation
def np_array_to_list(img, w, h):
    #img = img.reshape(img.size,)
    #return img
    start_time = time.time()
    #tmp = [img[i][j] for i in range(h) for j in range(w)]
    tmp = img.reshape(img.size)
    #print "It takes %.4f s to convert to list." % (time.time() - start_time)
    return tmp

def list_to_np_array(ilist, w, h):
    start_time = time.time()
    """
    img = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            img[i][j] = ilist[i*w+j]
    print "It takes %.4f s to convert to array." % (time.time() - start_time)
    """
    img = ilist.reshape((w, h))
    return img
    #ilist = ilist.reshape((w, h))
    #return ilist

def get_image(path):
    img = imgio.read_img_uc(path)
    img = process.filter_scale(img, width=70, height=70)
    img = process.filter_threshold(img, threshold=128)
    # img = process.filter_erosion(img, window_size=5)
    return img

def spa_shapedistance(img1, img2):
    im1 = list_to_np_array(img1, 70, 70)
    im2 = list_to_np_array(img2, 70, 70)
    sc1 = ShapeContext(im1)
    sc2 = ShapeContext(im2)
    return ShapeContextMatcher(sc1, sc2).calculate_shape_distance()

def shapedistance(sc1, sc2):
    matching = ShapeContextMatcher(sc1, sc2)
    return matching.calculate_shape_distance()

def samelistdist(list1):
    start = time.time()
    dist = []
    for i in range(len(list1)):
        for j in range(i+1, len(list1)):
            dist.append(spa_shapedistance(list1[i], list1[j]))#shapedistance(list1[i], list1[j]))
    #print "Same list time:", time.time() - start
    return min(dist)
    #print "Same list time:", time.time() - start
    #dist = spa_dist.pdist(np.array(list1), metric=spa_shapedistance)
    #return dist.min()

def differentlistdis(list1, list2):
    dist = [spa_shapedistance(list1[i], list2[j]) for i in range(len(list1)) for j in range(len(list2))]
    #dist = [shapedistance(list1[i], list2[j]) for i in range(len(list1)) for j in range(len(list2))]
    return min(dist)
    #dist = spa_dist.cdist(np.array(list1), np.array(list2), metric=spa_shapedistance)
    #return dist.min()


time_1 = time.time()
folder_dir = 'annotated_captchas//train2'
training_set, training_labels = dataset.load_captcha_dataset(folder_dir)
model = SC_KNN_Decoder(dataset='test_easy_digits', character_shape=(70, 70))
time_2 = time.time()
print "Load label:", time_2 - time_1

"""
start_train_time = time.time()
model.fit(training_set, training_labels)
finish_train_time = time.time()
print "Train:", finish_train_time - start_train_time
#"""

start_test_time = time.time()
folder_dir = 'annotated_captchas//test2'
testing_set, testing_labels = dataset.load_captcha_dataset(folder_dir)

for index in range(190, len(testing_labels), 10):
    upper = min(index+2, len(testing_labels))
    model.fast_score(testing_set[index:upper], testing_labels[index:upper], mode='save', paras='fast_test'+str(index/10))

finish_test_time = time.time()
print "Test:", finish_test_time - start_test_time


