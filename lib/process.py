"""
Image Processing
"""

import numpy
from PIL import Image
from scipy.linalg import pascal
from lib import imgio as ImgIO



def _get_pixel_RGB(image, x, y):
    return (image[x][y][0], image[x][y][1], image[x][y][2])


def _set_pixel_RGB(image, x, y, r, g ,b):
    image[x][y][0] = r
    image[x][y][1] = g
    image[x][y][2] = b


def _filling_origin_color(image, old_color, x, y, r, g, b):
    next_r,next_g,next_b = _get_pixel_RGB(image, x, y)
    if old_color[0] == next_r and old_color[1] == next_g and old_color[2] == next_b:
        filter_flood_fill(image,x,y,r,g,b)


def _get_shape(image):
    img_height = len(image)
    img_width = len(image[0])
    return (img_width, img_height)


def _get_mode(image):
    mode = {1:'Gray', 3:'RGB', 4:'RGBA'}
    if image[0][0].size in mode.keys():
        return mode[image[0][0].size]
    else:
        return 'Invalid'


def get_background_color(image):
    color_index = numpy.zeros(256, dtype=numpy.int)
    for i in range(len(color_index)):
        color_index[i] = len(image[image == i])
    return numpy.argmax(color_index)


def gray_to_rgb(image):
    width, height = _get_shape(image)
    new_img = numpy.zeros((height, width, 3))
    for j in range(height):
        for i in range(width):
            new_img[j, i, 0] = new_img[j, i, 1] = new_img[j, i, 2] = image[j, i]
    return new_img



# def rgb_to_gray(image):
#     width, height = _get_shape(image)
#     new_img = numpy.zeros((height, width))
#     for j in range(height):
#         for i in range(width):
#             new_img[j, i] = (image[j, i, 0] + image[j, i, 1] + image[j, i, 2]) / 3.0
#     new_img = filter_threshold(new_img, threshold=254)
#     return new_img

def filter_threshold(image, threshold):
    width, height = _get_shape(image)
    new_img = image.copy()
    for j in range(height):
        for i in range(width):
            new_img[j, i] = 255.0 if image[j, i] >= threshold else 0.0
    return new_img


def filter_threshold_RGB(image, threshold):
    band = image[0][0].size
    if band == 1:
        new_img = filter_threshold(image, threshold)
    else:
        width, height = _get_shape(image)
        new_img = numpy.zeros((height, width))
        for j in range(height):
            for i in range(width):
                sum = 0
                for k in image[j, i]:
                    sum += k
                new_img[j, i] = 255.0 if sum >= threshold * band else 0.0
    return new_img


def filter_inverse(image):
    new_img = 255 - image
    return new_img


def filter_fix_broken_characters(image):
    width, height = _get_shape(image)
    new_img = image.copy()
    background = get_background_color(image)
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            if image[j - 1, i] != background and image[j + 1, i] != background:
                new_img[j, i] = image[j - 1, i]
            elif image[j, i - 1] != background and image[j, i + 1] != background:
                new_img[j, i] = image[j, i - 1]
    return new_img



def filter_reduce_lines(image, median=200):
    width, height = image.shape[1], image.shape[0]
    new_img = image.copy()
    background = get_background_color(image)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for j in range(height):
        for i in range(width):
            for x, y in directions:
                if j+x <= 0 or j+x >= height or i+y <= 0 or i+y >= width:
                    new_img[j, i] = background
                elif image[j+x, i+y] > median:
                    new_img[j, i] = (background + image[j+x, i+y]) / 2.0

    return new_img




def filter_fill_holes(image):
    width, height = _get_shape(image)
    new_img = image.copy()
    background = get_background_color(image)
    for j in range(1, height-1):
        for i in range(1, width-1):
            if image[j,i] == background and image[j-1,i] == image[j+1,i] == image[j,i-1] == image[j,i+1] != background:
                new_img[j,i] = image[j-1, i]
    return new_img


def filter_remove_dots(image):
    width, height = _get_shape(image)
    new_img = image.copy()
    background = image[0][0]
    for j in range(1, height-1):
        for i in range(1, width-1):
            if image[j,i] != background and image[j-1,i] == image[j+1,i] == image[j,i-1] == image[j,i+1] == background:
                new_img[j,i] = background
    return new_img


def filter_flood_fill(image, x, y, r=255, g=0, b=0):
    width, height = _get_shape(image)
    if 0 <= x < width and 0 <= y < height:
        old_r, old_g, old_b = _get_pixel_RGB(image,x,y)
        _set_pixel_RGB(image,x,y,r,g,b)

        _filling_origin_color(image,(old_r,old_g,old_b),x-1, y, r,g,b)
        _filling_origin_color(image,(old_r,old_g,old_b),x+1, y, r,g,b)
        _filling_origin_color(image,(old_r,old_g,old_b),x, y-1, r,g,b)
        _filling_origin_color(image,(old_r,old_g,old_b),x, y+1, r,g,b)
    else:
        pass


def filter_mean_smooth(image, window_size=3):
    #return image.filter(ImageFilter.SMOOTH)
    half_size = window_size / 2
    width, height = _get_shape(image)
    new_img = image.copy()
    for j in range(half_size, height-half_size):
        for i in range(half_size, width-half_size):
            total = 0
            for x in range(-half_size, half_size+1):
                for y in range(-half_size, half_size+1):
                    total += image[j+y, i+x]
            new_img[j,i] = total / (window_size * window_size)
    return new_img


def filter_median(image, window_size=3):
    half_size = window_size / 2
    width, height = _get_shape(image)
    new_img = image.copy()
    for j in range(half_size, height-half_size):
        for i in range(half_size, width-half_size):
            pixels = []
            for x in range(-half_size, half_size+1):
                for y in range(-half_size, half_size+1):
                    pixels.append(image[j+y, i+x])
            #new_img[j,i] = total / (window_size * window_size)
            pixels.sort()
            new_img[j, i] = pixels[window_size * window_size / 2 + 1]
    return new_img


def filter_remove_confusion(image, threshold, window_size=3):
    half_size = window_size / 2
    width, height = _get_shape(image)
    new_img = image.copy()
    for j in range(half_size, height-half_size):
        for i in range(half_size, width-half_size):
            total = 0
            for x in range(-half_size, half_size+1):
                for y in range(-half_size, half_size+1):
                    total += image[j+y, i+x]
            if total < threshold:
                new_img[j,i] = total / (window_size * window_size)
                #print total / (window_size * window_size)
    return new_img



# def filter_erosion(image, window_size=3):
#     half_size = window_size / 2
#     width, height = _get_shape(image)
#     new_img = image.copy()
#     for j in range(half_size, height-half_size):
#         for i in range(half_size, width-half_size):
#             e = 1
#             for x in range(-half_size, half_size+1):
#                 for y in range(-half_size, half_size+1):
#                     e = e and (255 - image[j+y, i+x])
#             if e:
#                 new_img[j,i] = 0.0
#             else:
#                 new_img[j,i] = 255.0
#                 #print total / (window_size * window_size)
#     return new_img
#
#
# def filter_dilation(image, window_size=3):
#     half_size = window_size / 2
#     width, height = _get_shape(image)
#     new_img = image.copy()
#     for j in range(half_size, height-half_size):
#         for i in range(half_size, width-half_size):
#             e = 1
#             for x in range(-half_size, half_size+1):
#                 for y in range(-half_size, half_size+1):
#                     e = e and image[j+y, i+x]
#             if e:
#                 new_img[j,i] = 255.0
#             else:
#                 new_img[j,i] = 0.0
#                 #print total / (window_size * window_size)
#     return new_img


"""
TODO: some basic operation
"""

def filer_reduce_mesh(image, target=0, remove_color=255, ratio=0.8):
    height, width = image.shape
    new_img = image * 255.0 if image.max() <= 1.0 else image
    for i in range(height):
        if numpy.sum(image[i, :]==target) > width * ratio:
            new_img[i, :] = remove_color
    for j in range(width):
        if numpy.sum(image[:, j]== target) > height * ratio:
            new_img[:, j] = remove_color
    return new_img


def filter_remove_lines(image, background=255, foreground=0, ratio=0.2):
    if image.max <= 1.0:
        background /= 255.
        foreground /= 255.
    height, width = image.shape[0], image.shape[1]
    img = image.copy()
    for i in range(height):
        tmp = []
        for j in range(width):
            if image[i, j] == foreground:
                tmp.append(j)
            else:
                if len(tmp) > width * ratio:
                    for k in tmp:
                        img[i, k] = background
                tmp = []
    for j in range(width):
        tmp = []
        for i in range(height):
            if image[i, j] == foreground:
                tmp.append(i)
            else:
                if len(tmp) > height * ratio:
                    for k in tmp:
                        img[k, j] = background
                tmp = []
    return img

def filter_fill_border_background(image, background=255):
    height, width = image.shape[0], image.shape[1]
    if image.max() <= 1.0 and background > 1:
        background /= 255
    tmp = image
    tmp[0, :] = background
    tmp[height-1, :] = background
    tmp[:, 0] = background
    tmp[:, width-1] = background
    return tmp



def filter_scale(image, width=40, height=55):
    im = Image.fromarray(numpy.uint8(image))
    out = im.resize((width, height))
    new_img = numpy.array(out)
    return new_img



def filter_average_image(image_list):
    t = sum(image_list) / float(len(image_list))
    return t

def get_gaussker(window_size=9):
    g = 2 ** (1 - window_size) * numpy.diag(numpy.fliplr(pascal(window_size)))
    return numpy.array([g]) * numpy.array([g]).T
