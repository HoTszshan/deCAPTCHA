import lib.imgio as ImgIO
import os
import numpy as np
from lib import dataset
import re
import time
file_name_1 = 'data/gdgs/1.jpg'
file_name_2 = 'data/zhihu/3a8r-0.jpg'

test_list = [file_name_1, file_name_2]


def estimate_function_time(function, n_iter=1, input_list=None):
    start_time = time.time()
    if input_list:
        result = map(lambda n:map(function, input_list), range(n_iter))
    else:
        result = map(function, range(n_iter))
    finish_time = time.time()
    print ("It takes %.4f s to test %s function %d times." % ((finish_time - start_time), function.__name__, n_iter))
    return filter(lambda x:not x == None, result)[0] if len(filter(lambda x:not x == None, result)) > 0 else None

"""
ImgIO.show_images_list([ImgIO.read_web_image('http://login.sns.hongxiu.com/reg/CheckCode.aspx?r=631'),
                       ImgIO.read_web_image_uc('http://login.sns.hongxiu.com/reg/CheckCode.aspx?r=631'),
                        ImgIO.read_image(test_list[0]), ImgIO.read_image(test_list[1])])
"""

#dataset.generate_captcha_dataset(length=3, total=1000)
#print len(dataset.get_image_files('data/generate data'))

#dataset.download_captcha_images(URL='http://login.sns.hongxiu.com/reg/CheckCode.aspx?r=631', total=100)
"""
def t():
    return ImgIO.read_web_image('http://login.sns.hongxiu.com/reg/CheckCode.aspx?r=631')

for i in range(2, 16):
    img_list = map(lambda x: t(), range(i))
    ImgIO.show_images_list(img_list)
"""
"""
images1, labels1 = dataset.load_captcha_dataset('gimpy-r-ball', other_file=True)
images2, labels2 = dataset.load_captcha_dataset('ez-gimpy')
images3, labels3 = dataset.load_captcha_dataset('captcha_difficult', label_file='solutions.txt')
images4, labels4 = dataset.generate_captcha_dataset(target_dir='new_one')
"""

def tmp(num):
    i1, l1 = dataset.load_captcha_pkl('gimpy-r-ball')
    i2, l2 = dataset.load_captcha_pkl('ez-gimpy')
    i3, l3 = dataset.load_captcha_pkl('captcha_difficult')
    i4, l4 = dataset.load_captcha_pkl('new_one')
    ti, tl, tei, tli = dataset.stratified_shuffle_split(i1, l1, save_dir='gim')
    print len(tl), len(tli)
    #for index, (image, label) in enumerate(zip(i1, l1)):
    #    print index, image.shape, label
    """
    ImgIO.show_images_list(images1[:10])
    ImgIO.show_images_list(i1[:10])
    ImgIO.show_images_list(images2[:5])
    ImgIO.show_images_list(i2[:5])
    ImgIO.show_images_list(images3[:10])
    ImgIO.show_images_list(i3[:10])
    ImgIO.show_images_list(images4[:10])
    ImgIO.show_images_list(i4[:10])
    """

    #print len(dataset.get_single_label_unique(l1))

estimate_function_time(tmp)