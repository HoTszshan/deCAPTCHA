"""
IO operation for image files
"""
import cStringIO
import numpy as np
import urllib2
from PIL import Image
from matplotlib import pyplot as plt
from skimage import io
from functools import reduce

def read_image(image_path):
    image = io.imread(image_path).astype('f')#np.array(Image.open(image_path),'f')
    return image

def read_image_uc(image_path):
    image = np.array(Image.open(image_path).convert('L'),'f')
    return image

def write_image(image, new_path):
    img = image * 255.0 if image.max() <= 1.0 else image
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.save(new_path)

def write_image_uc(image, new_path):
    if image.ndim <= 2:
        write_image(image, new_path)
    else:
        img = np.mean(image, axis=2)
        write_image(img,new_path)


def show_image(image, title_name='Image',interpolation=None, nearest=False):
    plt.figure()#num='astronaut', figsize=(8,8))
    if nearest: interpolation = 'nearest'
    plt_image = image * 255.0 if np.max(image) <= 1.0 else image
    plt.imshow(np.uint8(plt_image), interpolation)
    plt.title(title_name)
    if plt_image.ndim <= 2:
        plt.gray()
    plt.axis('off')
    plt.show()

def show_images_list(img_list, title_names=None, interpolation=None, nearest=False):
    gray_flag = False
    plt.figure()#figsize=(10,10))#num='astronaut', figsize=(8,8))
    number = len(img_list)
    col, row = __get_n_row_col(number)
    if nearest: interpolation = 'nearest'
    for num in range(number):
        plt.subplot(row, col, num+1)
        title_name = title_names[num] if title_names else 'Image' + ' ' + str(num)
        plt.title(title_name)
        if img_list[num].ndim <= 2 and not gray_flag:
            plt.gray()
            gray_flag = True
        #print img_list[num].shape, type(img_list[num]), np.max(img_list[num]), img_list[num].dtype
        tmp_image = img_list[num].copy() * 255.0 if np.max(img_list[num]) <= 1.0 else img_list[num].copy()
        plt.imshow(np.uint8(tmp_image), interpolation=interpolation)
        plt.axis('off')
    plt.show()


def read_web_image(web_URL):
    content = __get_web_content(web_URL)
    content = cStringIO.StringIO(content)
    return io.imread(content).astype('f')

def read_web_image_uc(web_URL):
    content = __get_web_content(web_URL)
    content = cStringIO.StringIO(content)
    return read_image_uc(content)


def print_image_array(image):
    if image.ndim < 2:
        print image
    elif image.ndim == 2:
        for row in image:
            for col in row:
                print '%.4f'%col, '\t',
            print '%'
    else:
        for i in range(image.ndim):
            for row in image[:,:,i]:
                for col in row:
                    print '%.4f'%col, '\t',
                print '%'
            print '%' * image[:,:,i].shape[1]
    print image.shape


def __get_web_content(web_URL):
    userAgent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = { 'User-Agent' : userAgent }
    try:
        request = urllib2.Request(web_URL, headers=headers)
        response = urllib2.urlopen(request)
        return response.read()
    except urllib2.HTTPError, e:
        raise ValueError(e.code)
    except urllib2.URLError, e:
        raise ValueError(e.reason)

def __get_n_row_col(number):
    upper = np.ceil(np.sqrt(number))
    below = np.floor(np.sqrt(number))
    while upper * below < number:
        upper += 1
    return upper, below

def __save_plot_image(title):
    plt.savefig(title + '.jpg')





# TODO: Histogram
