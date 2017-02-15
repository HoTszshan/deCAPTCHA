from lib import imgio
from lib import dataset
from lib import process
from lib.segment import CharacterSeparator
from feature.shapecontext import *

import os
import random
import math
import numpy as np
import numpy.linalg as nl

from sc_knn_decoder import *
import sys
sys.setrecursionlimit(1500)

def get_image(path):
    img = imgio.read_img_uc(path)
    img = process.filter_scale(img, width=70, height=70)
    img = process.filter_threshold(img, threshold=128)
    return img

def get_fast_prune_tag(fast_pruner, image, r_paras=0.3, threshold=1.00, cut_off=0.78):
    img_rsc = GeneralizedShapeContext(image, sample="rsc", sample_params=r_paras)
    tmp_list, voting_sorted = fast_pruner.get_voting_result(img_rsc, threshold, cut_off)
    return (tmp_list, voting_sorted)



"""
aver_known_images, aver_known_labels = pickle.load(open("average_image.pkl", "rb"))
kmed_known_images, kmed_known_labels = pickle.load(open("k-medoids.pkl", "rb"))

aver_known_labels = [label + 'a' for label in aver_known_labels]
kmed_known_labels = [label + 'k' for label in kmed_known_labels]
kmed_known_labels.extend(aver_known_labels)
kmed_known_images.extend([process.filter_threshold_RGB(img, threshold=150) for img in aver_known_images])
        #aver_known_images)

known_images_gsc = [GeneralizedShapeContext(image) for image in kmed_known_images]#aver_known_images]
fast_pruner = FastPruner(kmed_known_labels, known_images_gsc)
"""
"""
file_list = ['test_easy_digits/1/32.jpg', 'test_easy_digits/3/35.jpg']
img_list = []
for path in file_list:
    img_list.append(ImgIO.read_img_uc(path))
ImgIO.show_img_list(img_list)
"""
"""
for img in img_list:
    #img = process.filter_threshold_RGB(img, threshold=150)
    #imgio.show_img(img)
    sc = ShapeContext(img)
    sc.display_norm_points(sc.norm_sample_points)
#"""


model_file = 'test_easy_digits' + '/' + 'model' + '.pkl'
fast_prune_file = 'test_easy_digits' + '/' + 'fast_pruner' + '.pkl'
prune_engine_file = 'test_easy_digits' + '/' + 'fast_model'

fast_pruner = pickle.load(open(fast_prune_file, "rb")) if os.path.isfile(fast_prune_file) else None
fast_engine = dir_archive(prune_engine_file, {}, serialized=True)
fast_engine.load()
engine = pickle.load(open(model_file, "rb"))

"""
if os.path.isfile(fast_prune_file):
    fast_pruner = pickle.load(open(fast_prune_file, "rb"))
    #self.fast_pruner, self.fast_engine = pickle.load(open(fast_model_file, "rb"))
    if os.path.exists(prune_engine_file):
        fast_engine = dir_archive(prune_engine_file, {}, serialized=True)
        fast_engine.load()
if os.path.isfile(model_file):
    engine = pickle.load(open(model_file, "rb"))
#"""


#"""
test_path = 'annotated_captchas/test2/9956-0.jpg'
label = test_path.split('/')[-1]
label = label.split('-')[0]

image = ImgIO.read_img_uc(test_path)

separator = CharacterSeparator(pre_processing_digit(image), (70, 70))
img_list = separator.segment_process()
char_testing_list = [np_array_to_list(c) for c in img_list]
print  engine.predict(char_testing_list)

#ImgIO.show_img_list(c_list)


#"""

#for path in file_list:
#    image = ImgIO.read_img_uc(path)
for image, key in zip(img_list,label):
    tag, sort_dict = get_fast_prune_tag(fast_pruner, image)
    #print 'Path:',path, '\t', 'Tag:',''.join(tag),
    #key = path.split('/')[1]
    sc = ShapeContext(image)
    #sc.display_norm_points(sc.norm_sample_points)
    print "Key: ", key, "Tag: ", ''.join(tag),
    if key in tag:#str(key)+'a' in tag or str(key)+'k' in tag:
        print ' '
    else:
        print '\t\t',key, 'is not in tag!!'
        for label, voting in sort_dict:
            print label, '\t', voting
    tag_str = ''.join(tag)
    if tag_str in fast_engine.keys():
        print "Predict Result:", fast_engine[tag_str].predict([np_array_to_list(image)])
    else:
        print "Not in fast keys! Predict Result:", engine.predict([np_array_to_list(image)])

#"""

folder = 'test_easy_digits' #'test_digits'
digits = {}.fromkeys(np.linspace(0, 9, 10, dtype=np.int))
del digits[7]


wrong_keys = []
long_tags = []
keys = []
single_key = []

for key in digits.keys():
    dir_path = folder + '/' + str(key)
    print ' '
    print dir_path
    file_list = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.endswith('.jpg')]

    for image_path in file_list:
        image = imgio.read_img_uc(image_path)
        tag, sort_dict = get_fast_prune_tag(fast_pruner, image)
        print 'Path:', image_path, '\t', 'Tag:',''.join(tag),
        if str(key) in tag:#str(key)+'a' in tag or str(key)+'k' in tag:
            print ' '
        else:
            print '\t\t',key, 'is not in tag!!'
            wrong_keys.append(image_path)
            for label, voting in sort_dict:
                print label, '\t', voting
        """
        if len(tag) >= 7 and str(key) in tag:
            long_tags.append(image_path)
            for label, voting in sort_dict:
                print label, '\t', voting
            #sc = ShapeContext(image)
            #sc.display_norm_points(sc.norm_sample_points)
        #"""
        if len(tag) < 7 and not ''.join(tag) in keys:
            keys.append(''.join(tag))
        if len(tag) <= 2 and not ''.join(tag) in single_key:
            single_key.append(''.join(tag))



print "Wrong prune:", len(wrong_keys), "Wrong keys:", wrong_keys
print "Too long:", len(long_tags), "Long images:", long_tags
print '\n', 'Keys:', keys
print "Single key:", single_key