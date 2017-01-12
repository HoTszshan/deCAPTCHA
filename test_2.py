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



def get_image(path):
    img = imgio.read_img_uc(path)
    img = process.filter_scale(img, width=70, height=70)
    img = process.filter_threshold(img, threshold=128)
    return img
#for i in range(10):
img_file_1 = 'digit\\0\\10.jpg'
img_file_2 = 'digit\\0\\15.jpg' # + str(i) + '\\15.jpg'
img_1 = get_image(img_file_1)
img_2 = get_image(img_file_2)

tmp_time_1 = time.time()
sc_1 = ShapeContext(img_1)
sc_2 = ShapeContext(img_2)
tmp_time_2 = time.time()
matcher = ShapeContextMatcher(sc_1, sc_2)
matcher.calculate_shape_distance(l_0=1000)

tmp_time_3 = time.time()


tmp_time_n1 = time.time()
t_sc_1 = shapecontext_1.ShapeContext(img_1)

t_sc_2 = shapecontext_1.ShapeContext(img_2)
tmp_time_n2 = time.time()
t_matcher = shapecontext_1.ShapeContextMatcher(t_sc_1, t_sc_2)
t_matcher.calculate_shape_distance()


tmp_time_n3 = time.time()

tmp_time = time.time()
gsc_1 = GeneralizedShapeContext(img_1, sample='rsc', sample_params=0.3)
print "Generalized Shape Context Time:", time.time() - tmp_time
#gsc_2 = GeneralizedShapeContext(img_2)

print "Construct original sc:", tmp_time_n2 - tmp_time_n1, '\t',
print "Original matching time:", tmp_time_n3 - tmp_time_n2, '\t',
print "Total time:", tmp_time_n3 - tmp_time_n1

print "Construct updated sc:", tmp_time_2 - tmp_time_1, '\t',
print "Updated matching time:", tmp_time_3 - tmp_time_2, '\t',
print "Total time:", tmp_time_3 - tmp_time_1
#"""
#print '\n\n\n'

