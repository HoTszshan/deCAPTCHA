# extract simple and basic features
from feature import process2
from lib import imgio as ImgIO
import numpy as np
from shapecontext import ShapeContext
from skimage import measure


class ScaleExtract(object):
    def __init__(self, extract_func):
        self.callback = extract_func

    def __scale_down(self, image):
        # ImgIO.show_image(process2.resize_transform(image, output_shape=(16,16)))
        return process2.resize_transform(image, output_shape=(16,16))

    def __call__(self, image, features):
        return self.callback(self.__scale_down(image), features, prefix='scaled-')



class SkeletonExtract(object):
    def __init__(self, extract_func):
        self.callback = extract_func

    def __extract_skeleton(self, image):
        return process2.extract_skeleton(image)

    def __call__(self, image, features):
        return self.callback(self.__extract_skeleton(image), features, prefix='scaled-')



# Extract function

def position_brightness(image, features, prefix=''):
    height, width = image.shape[0], image.shape[1]
    for y in range(width):
        for x in range(height):
            features[prefix+'pos-'+str(x*height+y)] = image[x, y]

def line_histogram(image, features, prefix='', foreground=None, axis=0):
    tag = 'col-' if axis == 0 else 'row-'
    foreground = foreground if foreground else process2.get_background_color(image, foreground=True)
    histogram = process2.pixels_histogram(image, axis=axis, color=foreground)
    for i in range(len(histogram)):
        features[prefix+tag+str(i)] = histogram[i]

def pixel_count(image, features, prefix='', foreground=None):
    features[prefix+'count'] = process2.get_pixel_count(image, foreground)

def shape_contest(image, features, prefix='', sample_num=100):
    sc = ShapeContext(image, sample_num=100)
    points = [(i,sc.sample_points[i]) for i in range(len(sc.norm_sample_points))]
    tmp = sorted(points, key=lambda x:x[1][0]*100+x[1][1])
    for index, point in tmp:
        for i in range(len(sc.sc_dict[index])):
            features[prefix+'sc-'+str(index)+'-'+str(i)] = sc.sc_dict[index][i]

def hu_moments(image, features, prefix=''):
    height, width = image.shape[0], image.shape[1]
    mu = measure.moments_central(image, height*0.5, width*0.5)
    nu = measure.moments_normalized(mu)
    hu = measure.moments_hu(nu)
    for i in range(len(hu)):
        features[prefix+'hu-'+str(i)] = hu[i]


def points13(image, features, prefix='', foreground=None):
    if foreground == None:
        foreground = process2.get_background_color(image, foreground=True)
    height, width = image.shape[0], image.shape[1]
    row_coor = np.linspace(0, height, 5, dtype=np.int8)
    col_coor = np.linspace(0, width, 3, dtype=np.int8)
    for i in range(4):
        for j in range(2):
            pixels = process2.get_pixel_count(image[row_coor[i]:row_coor[i+1], col_coor[j], col_coor[j+1]], foreground=foreground)
            features[prefix+'p13-'+str(i*2+j)] = pixels
    row_index = np.linspace(0, height-1, 4, dtype=np.int8)
    col_index = np.linspace(0, width-1, 4, dtype=np.int8)
    features[prefix+'p13-8'] = sum(image[row_index[1], :]==foreground)
    features[prefix+'p13-9'] = sum(image[row_index[2], :]==foreground)
    features[prefix+'p13-10'] = sum(image[:, col_index[1]]==foreground)
    features[prefix+'p13-11'] = sum(image[:, col_index[2]]==foreground)
    features[prefix+'p13-12'] = process2.get_pixel_count(image, foreground=foreground)


def coarse_mesh_count(image, features, prefix='', foreground=None, size=4):
    if foreground == None:
        foreground = process2.get_background_color(image, foreground=True)
    height, width = image.shape[0], image.shape[1]
    row_coor = np.linspace(0, height, size+1, dtype=np.int8)
    col_coor = np.linspace(0, width, size+1, dtype=np.int8)
    for i in range(size):
        for j in range(size):
            pixels = process2.get_pixel_count(image[row_coor[i]:row_coor[i+1], col_coor[j]:col_coor[j+1]], foreground=foreground)
            features[prefix+'mesh-'+str(i)+'-'+str(j)] = pixels
    # for key, item in features.items():
    #     print key, item
    # print '\n\n'

