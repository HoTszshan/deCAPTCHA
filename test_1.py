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

start_time = time.time()
folder = 'digits'
#digits = {}.fromkeys(np.linspace(0, 9, 10, dtype=np.int))
#del digits[7]

"""
imgs = []
labels = []
"""
"""
loading_time = time.time()
for num in digits.keys():
    digits[num] = []
    img_folder = folder + '\\' + str(num)
    for i in range(10):
        file_path = img_folder + '\\' + str(i) + '.jpg'
        img = get_image(file_path)
        # sc = ShapeContext(img)
        digits[num].append(img)#np_array_to_list(img, 70, 70))#sc)
"""
"""
        imgs.append(np_array_to_list(img, 70, 70))
        labels.append(str(num))

"""

"""
for i in digits.keys():
    t1 = time.time()
    min_cost_inlist = samelistdist(digits[i])
    t2 = time.time()
    print i, min_cost_inlist, '\t\t'
    print 'T(min cost in list) :', t2 - t1, 's,  '
    min_cost_list = []
    for j in digits.keys():
        if not i == j:
            t = differentlistdis(digits[i], digits[j])
            min_cost_list.append(t)
            print (i, j), t, '\t',
    t3 = time.time()
    print 'T(min cost in diff list) :', (t3 - t2) / 60.0, 'min'
#"""
"""
print "Loading time:", time.time() - loading_time
nbrs = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree', metric=spa_shapedistance)
start_training = time.time()
nbrs.fit(imgs, labels)
print "Training time:", time.time() - start_training
"""

known_images, known_labels = pickle.load(open("k-medoids.pkl", "rb"))
#known_images, known_labels = pickle.load(open("average_image.pkl", "rb"))
#a_imgs = [process.filter_average_image(digits[i]) for i in digits.keys()]
#known_images = [process.filter_threshold_RGB(img, 140) for img in known_images]
#ImgIO.show_img_list(known_images)

gsc_list = [GeneralizedShapeContext(img) for img in known_images]
# for i in range(len(known_labels)):
#    gsc_list[i].display_r_points(gsc_list[i].norm_sample_points)

fast_pruner = FastPruner(known_labels, gsc_list)
load_time = time.time()
"""
#for j in range(10):
rsc_list = [GeneralizedShapeContext(digits[num][j], sample='rsc', sample_params=0.1) for num in digits.keys()]
r_label = [num for num in digits.keys()]
for i in range(len(r_label)):
    print r_label[i]
    fast_pruner.get_voting_result(rsc_list[i], 1.0)

    print '\n\n'

"""

print "It use %.6f s to prepare fast pruner." % (load_time - start_time)
"""
bed_result = []
bed_result_label = []
cut_off = []
bad = []
not_cut = []
train_engine = []
for i in range(len(known_labels)):
    num = known_labels[i]
    digit_folder = folder + '\\' + str(num)
    image_files = dataset._get_jpg_list(digit_folder)
    for image_path in image_files:
        print num
        image = get_image(image_path)
        img_rsc = GeneralizedShapeContext(image, sample="rsc", sample_params=0.30)
        tmp_list = fast_pruner.get_voting_result(img_rsc, 1.0052, 0.79)
        print tmp_list,
        if num in tmp_list:
            print "Yes!"
            if len(tmp_list) <= 2:
                # ImgIO.show_img_list([image, known_images[i]])#a_imgs[num]])
                cut_off.append(known_labels[i])
            if len(tmp_list) <= 3:
                tmp_str = ''.join(tmp_list)
                if not tmp_str in train_engine:
                    train_engine.append(tmp_str)
        else:
            print "No!"
            if len(tmp_list) <= 3:
                ImgIO.show_img_list([image, known_images[i]])#a_imgs[num]])
            # if len(tmp_list) >= 5:
            #    ImgIO.show_img_list([image, known_images[i]])#a_imgs[num]])
            bed_result.append(image_path)
            bed_result_label.append(known_labels[i])
        if len(tmp_list) == 1 and not num in tmp_list:
            bad.append(known_labels[i])
            ImgIO.show_img_list([image, known_images[i]])
        if len(tmp_list) >= 5:
            not_cut.append(tmp_list)
        print "\n\n\n"
print "It use %.5f min to test..." % ((time.time() - load_time) / 60.0)
print "bad_label", len(bed_result), bed_result_label
print "bad_label_path", bed_result
print "cut off label(yes)", cut_off
print "bad cut off:", bad
print "not cut:", len(not_cut)
print "To train:", train_engine
"""

time_1 = time.time()
folder_dir = 'annotated_captchas//train1'
training_set, training_labels = dataset.load_captcha_dataset(folder_dir)
model = SC_KNN_Decoder(dataset='digits', character_shape=(70, 70), sys='XOS')
time_2 = time.time()
print "Load label:", time_2 - time_1
model.fit(training_set, training_labels)


