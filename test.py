import lib.imgio as ImgIO
import os
import numpy as np
from lib import dataset

file_name_1 = 'data/gdgs/1.jpg'
file_name_2 = 'data/zhihu/3a8r-0.jpg'

test_list = [file_name_1, file_name_2]

"""
for name in test_list:
    i1 = ImgIO.read_image(name)
    i1 /= 255
    i2 = ImgIO.read_image_uc(name)
    #ImgIO.show_images_list([i1, i2])
"""
"""
ImgIO.show_images_list([ImgIO.read_web_image('http://login.sns.hongxiu.com/reg/CheckCode.aspx?r=631'),
                       ImgIO.read_web_image_uc('http://login.sns.hongxiu.com/reg/CheckCode.aspx?r=631'),
                        ImgIO.read_image(test_list[0]), ImgIO.read_image(test_list[1])])
"""

dataset.generate_captcha_dataset(length=3, total=1000)
print len(dataset.get_image_files('data/generate data'))