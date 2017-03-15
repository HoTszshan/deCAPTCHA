from scipy import ndimage
from skimage import filters
from skimage.morphology import disk
from skimage import restoration, exposure, transform
from skimage import morphology
from lib import process
import numpy as np
import random



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
        count = sum(image[image==foreground])
    else:
        count = np.all(image[:,:] == foreground, axis=2).sum()
    return co


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

def threshold_filter(image, threshold=127):
    new_image = rgb_to_gray(image) if image.ndim > 2 else image
    result =  new_image > threshold
    return result.astype(image.dtype)

def inverse(image, background=255.0):
    return background - image



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
    detect_image = gray_to_rgb(image) if image.ndim == 2 else image.copy()
    target_image = gray_to_rgb(image) if image.ndim == 2 else image.copy()
    foreground = get_background_color(image, foreground=True)
    background = get_background_color(image)
    for x, y in get_coordinates_list(height, width):
        object_pixels = []
        color_fill_detect_objects(detect_image, object_pixels, x, y, target_color=foreground, N_4=n_4)
        if len(object_pixels) > 0:
            #print "Length of object list:" + str(len(object_pixels))
            if len(object_pixels) < remove_size:
                color_fill_update_image_by_object(target_image, object_pixels, background_color=background)
    return target_image

# extract skeleton
def extract_skeleton(image):
    new_image = rgb_to_gray(image) if image.ndim > 2 else image#.copy()
    new_image = max_min_normalization(new_image, min_value=0., max_value=1.) if new_image.max > 1.0 else new_image
    skeleton = morphology.skeletonize(new_image)
    return skeleton





def resize_transform(image, output_shape=(16,16)):
    new_image = image / 255.0 if image.max() > 1.0 else image
    return transform.resize(new_image, output_shape)

def scale_transform(image, ratio=0.5):
    return transform.rescale(image, ratio)

def rotate_transform(image, angle=30.0, reshape=False):
    return transform.rotate(image, angle, reshape)


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



def normalise_scaling(image, output_shape=(40,40)):
    if not __is_binary(image):
        return resize_transform(image, output_shape=output_shape)
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
            return scale_transform(tmp, ratio=h_ratio)
        elif h_ratio < w_ratio:
            new_image = np.ones((output_shape)) * get_background_color(image)
            # norm_tmp = scale_transform(image, h_ratio)
            norm_tmp = process.filter_scale(tmp, height=int(round(height*h_ratio)), width=int(round(width*h_ratio)))
                #resize_transform(tmp, output_shape=(int(round(height*h_ratio)), int(round(width*h_ratio))))
            width_gap = int((new_image.shape[1] - norm_tmp.shape[1]) * 0.5)
            new_image[:, width_gap:(width_gap+norm_tmp.shape[1])] = norm_tmp[:,:]
            # ImgIO.show_images_list([tmp, norm_tmp, new_image])
            return new_image
        else:
            new_image = np.ones((output_shape)) * get_background_color(image)
            norm_tmp = process.filter_scale(image, height=int(round(height*w_ratio)), width=int(round(width*w_ratio)))
                #resize_transform(tmp, output_shape=(int(round(height*w_ratio)), int(round(width*w_ratio))))
            height_gap = int((new_image.shape[0] - norm_tmp.shape[0]) * 0.5)
            new_image[height_gap:(height_gap + norm_tmp.shape[0]), :] = norm_tmp[:, :]
            return new_image


def normalise_rotation(image, angle_range=(-30, 30)):
    def cal_width(img):
        tmp = otsu_filter(img)
        col_histogram = pixels_histogram(tmp, axis=0)
        col_index = filter(lambda x:not (col_histogram[x] == 0), range(len(col_histogram)))
        return col_index[-1] - col_index[0] + 1
    min_v, max_v = angle_range
    coarse_angles = np.linspace(min_v, max_v, 13)
    coarse_widths = map(lambda x:cal_width(rotate_transform(image, x, reshape=False)), coarse_angles)
    min_c_angle = coarse_angles[random.choice([i for i in range(len(coarse_widths)) if coarse_widths[i] == min(coarse_widths)])]

    finer_angles = np.linspace(min_c_angle-10, min_c_angle+10, 21)
    finer_widths = map(lambda x:cal_width(rotate_transform(image, x, reshape=False)), finer_angles)
    best_angle = finer_angles[random.choice([i for i in range(len(finer_widths)) if finer_widths[i] == min(finer_widths)])]
    print best_angle
    return rotate_transform(image, best_angle)



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



# denoised = filter.rank.median(image, morphology.disk(2)) #remove noise

################################
# Fix function is in process
################################
