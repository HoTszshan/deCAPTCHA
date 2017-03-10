from scipy import ndimage
from skimage import filters
from skimage.morphology import disk
from skimage import restoration, exposure, transform
from skimage import morphology
import numpy as np


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

def rgb_to_gray(image, **params):
    r = params['r'] if 'r' in params.keys() else None
    g = params['g'] if 'g' in params.keys() else None
    b = params['b'] if 'b' in params.keys() else None
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

def max_min_normalization(image, **params):
    min_value = params['min_value'] if 'min_value' in params.keys() else 0.0
    max_value = params['max_value'] if 'max_value' in params.keys() else 255.0
    min_exc = image.min()
    max_exc = image.max()
    def normalize(v, min_tar, max_tar, min_ori, max_ori):
        return (v - min_ori) * (max_tar - min_tar) / (max_ori - min_ori) + min_tar
    return np.array(map(lambda x: normalize(x, min_value, max_value, min_exc, max_exc), image))

def smooth(image, **params):
    # params: func, sigma, size
    if len(params.keys())==0 or (params['func'] == 'gaussian' and not 'sigma' in params.keys()):
        def smooth_func(image):
            return __i_filter['gaussian'](image, sigma=3)
    elif 'sigma' in params.keys():
        def smooth_func(image):
            return __i_filter['gaussian'](image, sigma=params['sigma'])
    elif 'size' in params.keys():
        def smooth_func(image):
            return __i_filter['uniform'](image, size=params['size'])
    elif (params['func'] == 'uniform' and not 'size' in params.keys()):
        def smooth_func(image):
            return __i_filter['uniform'](image, size=11)
    else:
        print('No define well and use gaussian function and gamma = 3')
        def smooth_func(image):
            return __i_filter['gaussian'](image, sigma=3)

    return smooth_func(image)

def sharpen(image, **params):
    # params: alpha, sigma
    alpha = params['alpha'] if 'alpha' in params.keys() else 30
    sigma = params['sigma'] if 'sigma' in params.keys() else 3
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
def sci_median(image, **params):
    size = params['size'] if 'size' in params.keys() else 3
    return ndimage.median_filter(image, size=size)

# good for circle
def sk_median(image, **params):
    new_img = rgb_to_gray(image) if image.ndim > 2 else image
    new_img = max_min_normalization(new_img, min_value=-1, max_value=1)
    return filters.rank.median(new_img, disk(1))

def denoise_tv(image, **params):
    weight = params['weight'] if 'weight' in params.keys() else 0.1
    return restoration.denoise_tv_chambolle(image, weight)

#  Histogram-based method: Otsu thresholding
def otsu_filter(image):
    new_image = rgb_to_gray(image) if image.ndim > 2 else image
    threshold_o = filters.threshold_otsu(image)
    result = new_image < threshold_o
    return result.astype(image.dtype)

def threshold_filter(image, **params):
    thre = params['threshold'] if 'threshold' in params.keys() else 127
    new_image = rgb_to_gray(image, **params) if image.ndim > 2 else image
    result =  new_image < thre
    return result.astype(image.dtype)

def inverse(image, **params):
    background = params['background'] if 'background' in params.keys() else 255.0
    return background - image



def erosion(image, **params):
    structure = params['structure'] if 'structure' in params.keys() else None
    #ndimage.binary_erosion(image).astype(image.dtype),
    #return morphology.binary_erosion(image, morphology.diamond(1)).astype(image.dtype)
    return ndimage.binary_erosion(image, structure=structure).astype(image.dtype) #np.ones((3,3)))

def dilation(image, **params):
    structure = params['structure'] if 'structure' in params.keys() else None
    return ndimage.binary_erosion(image, structure=structure).astype(image.dtype)

# opening: erosion + dilation
def opening(image, **params):
    structure = params['structure'] if 'structure' in params.keys() else None
    #return ndimage.binary_opening(image, structure=np.ones((3,3))).astype(image.dtype)
    return ndimage.binary_opening(image, structure=structure).astype(image.dtype)

# closing: dilation + erosion
def closing(image, **params):
    structure = params['structure'] if 'structure' in params.keys() else None
    return ndimage.binary_closing(image, structure=structure).astype(image.dtype)

# reconstruction (erosion + propagation) is better than opening/closing
def reconstruction(image, **params):
    structure = params['structure'] if 'structure' in params.keys() else None
    eroded_area = erosion(image, **params)
    return ndimage.binary_propagation(eroded_area, structure=structure, mask=image)

def __color_fill_detect_objects(img, object_pixel_list, row, col, target_color=np.array([0,0,0]),
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
                    __color_fill_detect_objects(img, object_pixel_list, row+x, col+y, target_color, filling_color, N_4)
        else:
            pass

def __color_fill_update_image_by_object(img, noise_object_list, background_color=np.array([255.,255.,255.])):
        for x, y in noise_object_list:
            img[x, y] = background_color

def denosie_color_filling(image, **params):
    remove_size = params['remove_size'] if 'remove_size' in params.keys() else 30
    n_4 = params['N_4'] if 'N_4' in params.keys() else False
    width, height = image.shape[1], image.shape[0]
    detect_image = gray_to_rgb(image) if image.ndim == 2 else image#.copy()
    target_image = gray_to_rgb(image) if image.ndim == 2 else image#.copy()
    foreground = get_background_color(image, foreground=True)
    background = get_background_color(image)
    for x, y in get_coordinates_list(height, width):
        object_pixels = []
        __color_fill_detect_objects(detect_image, object_pixels, x, y, target_color=foreground, N_4=n_4)
        if len(object_pixels) > 0:
            #print "Length of object list:" + str(len(object_pixels))
            if len(object_pixels) < remove_size:
                __color_fill_update_image_by_object(target_image, object_pixels, background_color=background)
    return target_image

# extract skeleton
def extract_skeleton(image):
    new_image = rgb_to_gray(image) if image.ndim > 2 else image#.copy()
    new_image = max_min_normalization(new_image, min_value=0., max_value=1.) if new_image.max > 1.0 else new_image
    skeleton = morphology.skeletonize(new_image)
    # ImgIO.show_image(new_image)
    #
    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    #
    # ax1.imshow(new_image, cmap=plt.cm.gray)
    # ax1.axis('off')
    # ax1.set_title('original', fontsize=20)
    #
    # ax2.imshow(skeleton, cmap=plt.cm.gray)
    # ax2.axis('off')
    # ax2.set_title('skeleton', fontsize=20)
    #
    # fig.tight_layout()
    # plt.show()
    return skeleton


def resize_transform(image, **params):
    new_shape = params['size'] if 'size' in params.keys() else (16, 16)
    new_image = image / 255.0 if image.max() >= 1.0 else image
    return transform.resize(new_image, new_shape)

def scale_transform(image, **params):
    ratio = params['ratio'] if 'ratio' in params.keys() else 1
    return transform.rescale(image, ratio)

def rotate_transform(image, **params):
    angle = params['angle'] if 'angle' in params.keys() else 0.0
    resize = params['resize'] if 'resize' in params.keys() else False
    return transform.rotate(image, angle, resize)

################################
# Fix function is in process
################################
