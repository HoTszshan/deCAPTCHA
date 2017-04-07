from scipy import ndimage
from skimage import filters
from skimage.morphology import disk
from skimage import restoration, exposure, transform
from skimage import morphology
from lib import process
import numpy as np
import random
import math
from scipy import signal

import lib.imgio as ImgIO
__i_filter = {'gaussian': ndimage.gaussian_filter, 'uniform': ndimage.uniform_filter}


def get_coordinates_list(n_row, n_col):
    x, y = np.meshgrid(np.arange(n_row), np.arange(n_col))
    return list(np.rec.fromarrays([x, y]).ravel())

def get_background_color(image, foreground=False):
    if image.ndim == 2:
        color_index = np.zeros(256, dtype=np.int)
        for i in range(len(color_index)):
            color_index[i] = len(image[image == i])
        return np.argmax(color_index) if not foreground else sorted([(i, value) for i, value in enumerate(color_index)],
                                                                    lambda x, y: cmp(x[1], y[1]))[-2][0]
    else:
        color_index = {}
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                color_index[tuple(image[row, col])] = 1 if not tuple(image[row,col]) in color_index.keys() else \
                    color_index[tuple(image[row, col])] + 1
        return np.array(max(color_index, key=color_index.get)) if not foreground else \
                np.array(list(sorted(color_index.items(), lambda x, y: cmp(x[1], y[1]))[-2][0]))

def get_pixel_count(image, foreground=None):
    foreground = foreground if foreground else get_background_color(image, foreground=True)
    if image.ndim == 2:
        count = (image==foreground).sum()
    else:
        count = np.all(image[:,:] == foreground, axis=2).sum()
    return count


################################
# Brightness filter
################################
def rgb_to_gray(image, r=None, g=None, b=None):
    if image.ndim == 2:
        return image
    else:
        if r and g and b:
            new_img = image.copy()
            new_img[:,:,0] = r * image[:,:,0]
            new_img[:,:,1] = g * image[:,:,1]
            new_img[:,:,2] = b * image[:,:,2]
            if image.shape[-1] == 4:
                new_img[:, :, 3] = 0 * image[:,:,3]
            return new_img.sum(axis=2)
        return image.mean(axis = 2)

def gray_to_rgb(image):
    if image.ndim == 2:
        new_image = image[:,:, np.newaxis]
        new_image = np.repeat(new_image, 3, axis=2)
        return new_image
    else:
        return image

def max_min_normalization(image, min_value=0.0, max_value=255.0):
    min_exc = image.min()
    max_exc = image.max()
    def normalize(v, min_tar, max_tar, min_ori, max_ori):
        return (v - min_ori) * (max_tar - min_tar) / (max_ori - min_ori) + min_tar
    return np.array(map(lambda x: normalize(x, min_value, max_value, min_exc, max_exc), image))

def median(image, kernel_size=None):
    return signal.medfilt(image, kernel_size=kernel_size)

def smooth(image, func='gaussian', sigma=3, size=0):
    # params: func, sigma, size
    if func == 'gaussian':
        def smooth_func(image):
            return __i_filter['gaussian'](image, sigma=sigma)
    elif func == 'uniform' and size == 0:
        def smooth_func(image):
            return __i_filter['uniform'](image, size=11)
    elif func == 'uniform' and size > 0:
        def smooth_func(image):
            return __i_filter['uniform'](image, size=size)
    else:
        print('No define well and use gaussian function and gamma = 3')
        def smooth_func(image):
            return __i_filter['gaussian'](image, sigma=3)

    return smooth_func(image)

def sharpen(image, alpha=30, sigma=3):
    def sharp_func(i):
        blurred_1 = __i_filter['gaussian'](i, sigma)
        blurred_2 = __i_filter['gaussian'](blurred_1, 1)
        new_img = blurred_1 + alpha * (blurred_1 - blurred_2)
        return new_img
    if image.ndim == 2:
        new_img = sharp_func(image)
    else:
        new_img = image#.copy()
        for i in range(image.shape[-1]):
            new_img[:,:, i] = sharp_func(image[:,:,i])
    return max_min_normalization(new_img)

def deconvolution(image):
    return signal.wiener(image)

# histogram equalization
def histogram_equalize(image):
    return exposure.equalize_hist(image)

# good for lines , corners
def sci_median(image, size=3):
    return ndimage.median_filter(image, size=size)

# good for circle
def sk_median(image):
    new_img = rgb_to_gray(image) if image.ndim > 2 else image
    new_img = max_min_normalization(new_img, min_value=-1, max_value=1)
    return filters.rank.median(new_img, disk(1))

def denoise_tv(image, weight=0.1):
    return restoration.denoise_tv_chambolle(image, weight)

#  Histogram-based method: Otsu thresholding
def otsu_filter(image):
    new_image = rgb_to_gray(image) if image.ndim > 2 else image
    threshold_o = filters.threshold_otsu(image)
    result = new_image > threshold_o
    return result.astype(image.dtype)


def yen_filter(image):
    new_image = rgb_to_gray(image) if image.ndim > 2 else image
    threshold_y = filters.threshold_yen(image)
    result = new_image > threshold_y
    return result.astype(image.dtype)

def threshold_filter(image, threshold=127):
    new_image = rgb_to_gray(image) if image.ndim > 2 else image
    result =  new_image > threshold
    return result.astype(image.dtype)

def threshold_RGB_filter(image, threshold=(20,20,20)):
    threshold = np.array(threshold)
    height, width = image.shape[0], image.shape[1]
    result = image[:,:] > threshold
    return result.astype(image.dtype)


def inverse(image):
    background = 1.0 if image.max() <= 1.0 else 255
    return background - image



################################
# morphology
################################
def erosion(image, structure=None):
    #structure = params['structure'] if 'structure' in params.keys() else None
    #ndimage.binary_erosion(image).astype(image.dtype),
    #return morphology.binary_erosion(image, morphology.diamond(1)).astype(image.dtype)
    return ndimage.binary_erosion(image, structure=structure).astype(image.dtype) #np.ones((3,3)))

def dilation(image, structure=None):
    return ndimage.binary_erosion(image, structure=structure).astype(image.dtype)

# opening: erosion + dilation
def opening(image, structure=None):
    #return ndimage.binary_opening(image, structure=np.ones((3,3))).astype(image.dtype)
    return ndimage.binary_opening(image, structure=structure).astype(image.dtype)

# closing: dilation + erosion
def closing(image, structure=None):
    return ndimage.binary_closing(image, structure=structure).astype(image.dtype)

# reconstruction (erosion + propagation) is better than opening/closing
def reconstruction(image, structure=None):
    eroded_area = erosion(image, structure)
    return ndimage.binary_propagation(eroded_area, structure=structure, mask=image)

# extract skeleton
def extract_skeleton(image):
    new_image = rgb_to_gray(image) if image.ndim > 2 else image#.copy()
    new_image = max_min_normalization(new_image, min_value=0., max_value=1.) if new_image.max > 1.0 else new_image
    skeleton = morphology.skeletonize(new_image)
    return skeleton



################################
# Color region
################################
def color_fill_detect_objects(img, object_pixel_list, row, col, target_color=np.array([0,0,0]),
                                filling_color=np.array([255.,0.,0.]), N_4=False):
        if np.all(img[row][col]== target_color):
            object_pixel_list.append((row,col))
            img[row, col] = filling_color
            if N_4:
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            else:
                directions = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if not (i == 0 and j == 0)]
            for x, y in directions:
                if 0 <= row+x < img.shape[0] and 0 <= col+y < img.shape[1]:
                    color_fill_detect_objects(img, object_pixel_list, row+x, col+y, target_color, filling_color, N_4)
        else:
            pass

def color_fill_update_image_by_object(img, noise_object_list, background_color=np.array([255.,255.,255.])):
        for x, y in noise_object_list:
            img[x, y] = background_color

def denosie_color_filling(image, remove_size=30, n_4=False):
    width, height = image.shape[1], image.shape[0]
    detect_image = gray_to_rgb(image) if image.ndim == 2 else image#.copy()
    target_image = gray_to_rgb(image) if image.ndim == 2 else image#.copy()
    foreground = get_background_color(image, foreground=True)
    background = get_background_color(image)
    for x, y in get_coordinates_list(height, width):
        object_pixels = []
        color_fill_detect_objects(detect_image, object_pixels, x, y, target_color=foreground, N_4=n_4)
        if len(object_pixels) > 0:
            #print "Length of object list:" + str(len(object_pixels))
            if len(object_pixels) < remove_size:
                color_fill_update_image_by_object(target_image, object_pixels, background_color=background)
    return rgb_to_gray(target_image)

# def detect_color_objects(image, n_4=False):
#     width, height = image.shape[1], image.shape[0]
#     detect_image = image.copy()
#     objects_list
def find_clutter_boundary(image, clutter_size=100, n_4=False):
    width, height = image.shape[1], image.shape[0]
    detect_image = gray_to_rgb(image) if image.ndim == 2 else image#.copy()
    foreground = get_background_color(image, foreground=True)
    boundarise_list = []
    for x, y in get_coordinates_list(height, width):
        object_pixels = []
        color_fill_detect_objects(detect_image, object_pixels, x, y, target_color=foreground, N_4=n_4)
        if len(object_pixels) > clutter_size:
            coor_mat = np.array(zip(*object_pixels))
            ceiling, floor = min(coor_mat[0,:]), max(coor_mat[0,:])
            lower, upper = min(coor_mat[1,:]), max(coor_mat[1,:])
            boundary = (ceiling, lower, floor, upper)
            boundarise_list.append(boundary)
    return boundarise_list

def remove_rect_block(image):
    col = pixels_histogram(image, axis=0)
    row = pixels_histogram(image, axis=1)
    def get_derivative(hist):
        der =  map(lambda x: hist[min(x+1, len(hist)-1)] - hist[x], range(len(hist)))
        tmp = []
        for i in range(len(der)):
            lower = max(0, i-1)
            upper = min(i+1, len(der)-1)
            if der[lower] - der[i] >= 10 and der[upper] - der[i] >= 10:
                tmp.append(i)
        if len(tmp) >= 2:
            return (tmp[0], tmp[-1])
        else:
            start, mid, end = int(len(hist) * 0.25), int(len(hist) * 0.5), int(len(hist)*0.75)
            left, right = [], []
            for i in range(start, mid):
                lower = max(0, i-1)
                upper = min(i+1, len(der)-1)
                if der[lower] - der[i] >= 0 and der[upper] - der[i] >= 0 and der[i] <0:
                    left.append((i, der[lower]-der[i]+der[upper]-der[i]))
            for i in range(mid, end):
                lower = max(0, i-1)
                upper = min(i+1, len(der)-1)
                if der[lower] - der[i] >= 0 and der[upper] - der[i] >= 0 and der[i] <0:
                    right.append((i, der[lower]-der[i]+der[upper]-der[i]))
            if left and right:
                left = sorted(left, key=lambda x: x[1])
                right = sorted(right, key=lambda x:x[1])
                left_cand, right_cand = left[-1][0], right[-1][0]
                # print "left: ",left
                # print "right: ", right
                return (left_cand, right_cand)
            else:
                return (int(len(hist) / 3.0), int(len(hist) / 3.0 * 2))
    def judge_coords(coor, height, width):
        x, y = coor
        # if x>=height/3. and x<=height*2./3. and y>=width/3. and y<= width*2./3.:
        if x>=height*0.25 and x<=height*0.75 and y>=width*0.25 and y<= width*0.75:
            return True
        else:
            return False
    lower, upper = get_derivative(col)
    ceiling, floor = get_derivative(row)
    new_image = image / 255.0 if image.max() >= 1.0 else image.copy()
    from skimage.feature import corner_harris,  corner_subpix, corner_peaks
    coords = corner_peaks(corner_harris(new_image), min_distance=13)
    if filter(lambda x: judge_coords(x, image.shape[0], image.shape[1]), coords):
        coords = np.array(filter(lambda x: judge_coords(x, image.shape[0], image.shape[1]), coords))
    try:
        ceiling_, floor_ = min(coords[:, 0]), max(coords[:, 0])
    except IndexError:
        print coords
    lower_, upper_ = min(coords[:, 1]), max(coords[:, 1])

    image_ = image.copy()
    l = min(lower, lower_) #lower_#
    c = min(ceiling, ceiling_) #ceiling_#
    u = max(upper, upper_)
    f = max(floor, floor_)
    if f-c>0 and u-l > 0:
        image_[c:f+1, l:u+1] = np.ones((f-c+1, u-l+1))*image.max() - image[c:f+1, l:u+1]
    return image_

def filling_background(image, n_4=False):
    width, height = image.shape[1], image.shape[0]
    detect_image = gray_to_rgb(image) if image.ndim == 2 else image#.copy()
    background = get_background_color(image, foreground=False)
    foreground = get_background_color(image, foreground=True)
    object_pixels = []
    color_fill_detect_objects(detect_image, object_pixels, 0, 0, target_color=background, N_4=n_4)
    print len(object_pixels), width*height
    if len(object_pixels) > width * height ** 0.5:
        color_fill_update_image_by_object(image, object_pixels, background_color=foreground)
    return rgb_to_gray(image)

def remove_interfering_arcs(image, threshold_range=(140, 180)):
    if image.max() <= 1.0:
        image *= 255
    def get_threshold_count(img, threshold):
        tmp = threshold_filter(img, threshold)
        return get_pixel_count(tmp, foreground=0)
    def get_best_threshold_image(img, t_range):
        # min_v, max_v = t_range
        # coarse_thresholds = np.linspace(min_v, max_v, int((max_v - min_v)/10)+1)
        # coarse_count = map(lambda x:get_threshold_count(img, x), coarse_thresholds)
        # min_mid_range = coarse_thresholds[random.choice([i for i in range(len(coarse_count)) if coarse_count[i] == min(coarse_count)])]
        # finer_thresholds = np.linspace(min_mid_range-5, min_mid_range+5, 11)
        # finer_count = map(lambda x:get_threshold_count(img, x), finer_thresholds)
        # best_threshold = finer_thresholds[random.choice([i for i in range(len(finer_count)) if finer_count[i] == min(finer_count)])]
        best_threshold = __get_best_min_value(img, t_range, get_threshold_count)
        return threshold_filter(img, threshold=best_threshold)
    tmp_image = get_best_threshold_image(image, threshold_range)
    tmp_image = median(tmp_image)
    tmp_image = median(tmp_image)
    interfering = inverse(tmp_image)
    target_image = otsu_filter(otsu_filter(image) + interfering)
    fore_color = get_background_color(target_image, foreground=True)
    tmp_list = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if interfering[i, j] and (target_image[max(0, i-1), j] == fore_color
                                      or target_image[min(image.shape[0]-1, i+1), j] == fore_color):
                # target_image[i, j] = fore_color
                tmp_list.append((i, j))
    for i, j in tmp_list:
        target_image[i,j] = fore_color

    return target_image

def gibbs_interfering_lines(image, threshold=150, background=None):
    def get_energy(area):
        e_filter = np.ones((3,3)) * 0.125
        e_filter[1,1] = 0
        return np.sum(area * e_filter)
    def get_remove_coordinate(img, t_threshold, b_color, i, j):
        h, w = img.shape[0], img.shape[1]
        if i-1<0 or j-1<0 or i+1==h or j+1 == w:
            return None
        energy = get_energy(img[i-1:i+2, j-1:j+2])
        if energy  > t_threshold and not img[i, j] == b_color:
            return (i, j)
    height, width = image.shape[0], image.shape[1]
    copy_image = image.copy()
    if not background:
        background = get_background_color(image)
    if image.max() <= 1:
        threshold /= 255.
    print threshold, background
    while True:
        # row, col = np.meshgrid(np.arange(height), np.arange(width))
        # row, col = row.reshape(height*width), col.reshape(height*width)
        # remove = map(lambda x, y:get_remove_coordinate(image, threshold, background, x, y), row, col)
        # remove = filter(lambda x: x, remove)
        remove = []
        for i in range(height):
            for j in range(width):
                if i-1<0 or j-1<0 or i+1==height or j+1==width:
                    # pass
                    if not image[i, j] == background:
                        remove.append((i, j))
                else:
                    energy = get_energy(image[i-1:i+2, j-1:j+2])
                    # print energy,
                    if energy > threshold and not image[i, j] == background:
                        remove.append((i, j))
        print '\n', len(remove), remove
        if len(remove) == 0:
            # ImgIO.show_images_list([copy_image, image])
            return image
        else:
            for coor in remove:
                image[coor] = background
            # map(lambda coor:image[coor]=background, remove)



################################
# Transformation
################################
def resize_transform(image, output_shape=(16,16)):
    new_image = image / 255.0 if image.max() > 1.0 else image
    return transform.resize(new_image, output_shape)

def scale_transform(image, ratio=0.5):
    return transform.rescale(image, ratio)

def rotate_transform(image, angle=30.0, reshape=False):
    return transform.rotate(image, angle, reshape)

def normalise_scaling(image, output_shape=(40,40),  background=None):
    if background == None:
        background = get_background_color(image)
    if not __is_binary(image):
        return process.filter_scale(image, height=output_shape[0], width=output_shape[1])
        # return resize_transform(image, output_shape=output_shape)
    else:
        target_height, target_width = output_shape
        min_row, min_col, max_row, max_col = get_object_boundaries(image)
        # print min_row, min_col, max_row, max_col
        width = max_col - min_col #+ 2
        height = max_row - min_row# + 2
        h_ratio, w_ratio = target_height / float(height), target_width / float(width)
        tmp = image[min_row:max_row+1, min_col:max_col+1]
        # print h_ratio, w_ratio
        if h_ratio == w_ratio:
            # ImgIO.show_images_list([image, tmp, scale_transform(tmp, ratio=h_ratio),
            #                         process.filter_scale(tmp, height=target_height, width=target_width)])
            return process.filter_scale(tmp, height=target_height, width=target_width)#scale_transform(tmp, ratio=h_ratio)
        elif h_ratio < w_ratio:
            new_image = np.ones((output_shape)) * background
            # norm_tmp = scale_transform(image, h_ratio)
            norm_tmp = process.filter_scale(tmp, height=int(round(height*h_ratio)), width=int(round(width*h_ratio)))
                #resize_transform(tmp, output_shape=(int(round(height*h_ratio)), int(round(width*h_ratio))))
            width_gap = int((new_image.shape[1] - norm_tmp.shape[1]) * 0.5)
            new_image[:, width_gap:(width_gap+norm_tmp.shape[1])] = norm_tmp[:,:]
            # ImgIO.show_images_list([tmp, norm_tmp, new_image])
            return new_image
        else:
            new_image = np.ones((output_shape)) * background
            norm_tmp = process.filter_scale(tmp, height=int(round(height*w_ratio)), width=int(round(width*w_ratio)))
                #resize_transform(tmp, output_shape=(int(round(height*w_ratio)), int(round(width*w_ratio))))
            height_gap = int((new_image.shape[0] - norm_tmp.shape[0]) * 0.5)
            new_image[height_gap:(height_gap + norm_tmp.shape[0]), :] = norm_tmp[:, :]
            # ImgIO.show_images_list([tmp, norm_tmp, new_image])
            return new_image

def normalise_rotation(image, angle_range=(-30, 30), min_width=True, axis=0):
    if image.max > 1.0:
        image /= 255.
    def cal_width(img, axis):
        tmp = otsu_filter(img)
        col_histogram = pixels_histogram(tmp, axis=axis)
        col_index = filter(lambda x:not (col_histogram[x] == 0), range(len(col_histogram)))
        return col_index[-1] - col_index[0] + 1
    min_v, max_v = angle_range
    coarse_angles = np.linspace(min_v, max_v, 13)
    coarse_widths = map(lambda x:cal_width(rotate_transform(image, x, reshape=False), axis=axis), coarse_angles)
    if min_width:
        min_c_angle = coarse_angles[random.choice([i for i in range(len(coarse_widths)) if coarse_widths[i] == min(coarse_widths)])]
        finer_angles = np.linspace(min_c_angle-10, min_c_angle+10, 21)
        finer_widths = map(lambda x:cal_width(rotate_transform(image, x, reshape=False), axis=axis), finer_angles)
        best_angle = finer_angles[random.choice([i for i in range(len(finer_widths)) if finer_widths[i] == min(finer_widths)])]
    else:
        max_c_angle = coarse_angles[random.choice([i for i in range(len(coarse_widths)) if coarse_widths[i] == max(coarse_widths)])]
        finer_angles = np.linspace(max_c_angle-10, max_c_angle+10, 21)
        finer_widths = map(lambda x:cal_width(rotate_transform(image, x, reshape=False), axis=axis), finer_angles)
        best_angle = finer_angles[random.choice([i for i in range(len(finer_widths)) if finer_widths[i] == max(finer_widths)])]
    # print best_angle
    return rotate_transform(image, best_angle)


def pixels_histogram(image, axis=0, foreground=True, color=None):
    target = get_background_color(image, foreground) if not color else color
    height, width = image.shape[0], image.shape[1]
    if axis == 0: # per col
        number_list = np.zeros(width)
        for i in range(len(number_list)):
            number_list[i] = sum([target == j for j in image[:, i]])
    else: # per row
        number_list = np.zeros(height)
        for i in range(len(number_list)):
            number_list[i] = sum([target == j for j in image[i, :]])
    return number_list

def boundary_histogram(image, axis=0, foreground=True, color=None):
    target = get_background_color(image, foreground) if not color else color
    height, width = image.shape[0], image.shape[1]
    if axis == 0: # per col
        number_list = np.zeros(width)
        for i in range(len(number_list)):
            col = filter(lambda x: x[1], zip(range(len(image[:, i])), image[:,i]==target))
            number_list[i] = col[-1][0] - col[0][0] if len(col)>=2 else 0
    else: # per row
        number_list = np.zeros(height)
        for i in range(len(number_list)):
            row = filter(lambda x: x[1], zip(range(len(image[i, :])), image[i,:]==target))
            number_list[i] = row[-1][0] - row[0][0] if len(row) >=2 else 0
    # from matplotlib import pyplot as plt
    # X = np.arange(len(number_list))
    # Y = np.array(number_list) / float(max(number_list))
    # plt.bar(X, +Y, facecolor='#9999ff', edgecolor='white')
    # for x, y in zip(X, Y):
    #     plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    # plt.ylim(0, +1.25)
    return number_list

def get_extremes_index(hist, minimum=True, offset=10, get_value=False):
    derivative =  map(lambda x: hist[min(x+1, len(hist)-1)] - hist[x], range(len(hist)))
    flag = 1 if minimum else -1
    extremes = []
    for i in range(len(derivative)):
        lower, upper  = max(0, i-1), min(i+1, len(derivative)-1)
        if (derivative[lower] - derivative[i]) * flag >= offset and (derivative[upper] - derivative[i]) * flag >= offset:
            extremes.append((i, derivative[i])) if get_value else extremes.append(i)
    if get_value:
        return zip(*extremes)
    else:
        return extremes

def get_object_boundaries(image):
    if not __is_binary(image):
        return 0, 0, image.shape[0], image.shape[1]
    else:
        row_histogram = pixels_histogram(image, axis=1)
        col_histogram = pixels_histogram(image, axis=0)
        def get_min_value(histogram):
            for i in  range(len(histogram)):
                if not histogram[i] == 0:
                    return i
                # if histogram[i]==0 and not histogram[i+1] == 0:
                #     return i
            else:
                return 0
        def get_max_value(histogram):
            for i in range(len(histogram)-1, 0, -1):
                if not histogram[i] == 0:
                    return i
                # if histogram[i]==0 and not histogram[i-1] == 0:
                #     return i
            else:
                return len(histogram)

        min_row, min_col = get_min_value(row_histogram), get_min_value(col_histogram)
        max_row, max_col = get_max_value(row_histogram), get_max_value(col_histogram)
        return min_row, min_col, max_row, max_col


def get_best_affine_angle(image, t_range=(45, 135)):
    foreground = get_background_color(image, foreground=True)
    if image.max() <= 1.0:
        min_r, max_r = t_range
        t_range = (min_r/255., max_r/255.)
    def get_shadow(image, angle, foreground=None):
        if not foreground: foreground =get_background_color(image, foreground=True)
        height, width = image.shape[0], image.shape[1]
        tmp = []
        for i in xrange(height):
            for j in xrange(width):
                if image[i][j] == foreground:
                    tmp.append(j-1.0*i/math.tan((math.radians(angle))))
        print foreground, tmp
        return max(tmp) - min(tmp)
    print foreground
    ans = __get_best_min_value(image, t_range=t_range, func=lambda img, x: get_shadow(img, x, foreground=foreground))
    return ans

def affine_image(image, angle=None):
    height, width = image.shape[0], image.shape[1]
    if not angle: angle = get_best_affine_angle(image)
    pts1=np.float32([[width/2,height/2],[width/2,0],[0,height/2]])
    pts2=np.float32([[width/2,height/2],[width/2+height/2/math.tan(math.radians(angle)),0],[0,height/2]])
    m = transform.estimate_transform(ttype='affine', src=pts1, dst=pts2)
    tmp_img = image / 255. if image.max() > 1.0 else image
    dst = transform.warp(tmp_img, m.inverse, output_shape=(height,width))
    # ImgIO.show_images_list([image, dst])
    return dst

def rotate_distort(image, factor):
    tmp_img = image / 255. if image.max() > 1.0 else image
    height, width = image.shape[0], image.shape[1]
    pts1 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    pts2 = np.float32([[factor, factor], [0, height-factor], [width-factor, 0], [height-factor, width-factor]])
    m = transform.estimate_transform(ttype='projective', src=pts1, dst=pts2)
    dst = transform.warp(tmp_img, m.inverse, output_shape=(height,width))
    if dst.max() <= 1. / 255.:
        dst = dst * 255
    return dst


def __is_binary(image):
    if image.ndim == 2:
        new_image = image * 255 if np.max(image) <= 1 else image
        color_index = np.zeros(256, dtype=np.int)
        for i in range(len(color_index)):
            color_index[i] = len(new_image[new_image == i])
        return True if len(color_index[color_index != 0]) <=2 else False
    else:
        color_index =  {}
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                color_index[tuple(image[row, col])] = 1 if not tuple(image[row,col]) in color_index.keys() else \
                    color_index[tuple(image[row, col])] + 1
        color_num = [color_index[i] for i in color_index.keys() if not color_index[i] == 0]
        return True if len(color_num) <= 2 else False


def __get_best_min_value(img, t_range, func):
    min_v, max_v = t_range
    coarse_values = np.linspace(min_v, max_v, int((max_v - min_v)/10)+1)
    coarse_count = map(lambda x:func(img, x), coarse_values)
    min_mid_range = coarse_values[random.choice([i for i in range(len(coarse_count)) if coarse_count[i] == min(coarse_count)])]
    finer_values = np.linspace(min_mid_range-5, min_mid_range+5, 11)
    finer_count = map(lambda x:func(img, x), finer_values)
    best_value = finer_values[random.choice([i for i in range(len(finer_count)) if finer_count[i] == min(finer_count)])]
    return best_value#threshold_filter(img, threshold=best_threshold)

# denoised = filter.rank.median(image, morphology.disk(2)) #remove noise

################################
# Fix function is in process
################################
def nochange(image):
    return image