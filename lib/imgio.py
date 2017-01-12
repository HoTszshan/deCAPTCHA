"""
IO operation for image files
"""
#import os
from PIL import Image
from numpy import *
#from pylab import *
from matplotlib import pyplot as plt


def read_img(imgname):
    img = array(Image.open(imgname),'f')
    return img

def read_img_uc(imgname):
    img = array(Image.open(imgname).convert('L'),'f')
    return img


def write_img(imgname, newpath):
    image = Image.fromarray(uint8(imgname))
    image.save(newpath)


def show_img(imgname, mode=0, title_name='Image'):
    plt.figure()#(num='astronaut', figsize=(8,8))
    plt.imshow(uint8(imgname))
    plt.title(title_name)
    if mode == 0:
        plt.gray()
    plt.axis('off')
    plt.show()


"""
TODO...
"""
def show_img_list(img_list, mode=0, title_name='Image'):
    plt.figure()#num='astronaut', figsize=(8,8))
    number = len(img_list)

    for num in range(number):
        plt.subplot(1,number,num+1)
        plt.title(title_name + ' ' + str(num))
        if mode == 0:
            plt.gray()
        plt.imshow(uint8(img_list[num]))
        plt.axis('off')

    plt.show()

