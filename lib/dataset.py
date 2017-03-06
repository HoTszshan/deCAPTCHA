"""
Get a dataset: Load, Generate, Download
"""

# import re
import os
import imgio
import random

from third_party.captcha.image import ImageCaptcha

FONTS_SET = ['/Library/Fonts/Arial.ttf', '/Library/Fonts/Brush Script.ttf', '/Library/Fonts/Phosphate.ttc']

# Get all file names of jpg images in the fold
def _get_jpg_list(dir_path):
    """
    Returns a list of filenames for all
    jpg images in a directory.
    :param path: fold path
    :return: a list of filepath
    """
    return [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.endswith('.jpg')]

def _get_png_list(dir_path):
    return [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.endswith('.png')]

def get_img_list(dir_path):
    return [os.path.join(dir_path,f) for f in os.listdir(dir_path)
            if f.endswith('.jpg')
            or f.endswith('.png')]


# The file name of the captcha is its responding label
def _get_image_lable(image_path):
    image = imgio.read_img_uc(image_path)
    # to get a label
    file_name = os.path.basename(image_path).split('.')[0]
    label = file_name.split('-')[0] if '-' in file_name else file_name
    #label = re.findall(r'^([0-9]+)-[0-9]+\..*$', file_name)[0]
    return (image,label)

def _initialize_digit_dictionary():
    dictionary = []
    for index in range(48,58):
        dictionary.append(chr(index))
    return dictionary

def _initialize_dictionary():
    dictionary = []
    for index in range(48,58):
        dictionary.append(chr(index))
    for index in range(65,91):
        dictionary.append(chr(index))
    for index in range(97, 123):
        dictionary.append(chr(index))
    return dictionary


# Load captcha dataset depend on the image fold path
def load_captcha_dataset(base_dir):
    files = get_img_list(base_dir)
    dataset = [ _get_image_lable(file_path) for file_path in files]
    #return dataset
    return zip(*dataset)

def generate_captcha_dataset(target_dir='generate data', total=1000, length=0, font_number=1):
    # Make a dir if not exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Set the font set
    gen_fonts = FONTS_SET[:]
    if 0 < font_number < len(FONTS_SET):
        #fonts = [fonts.pop(random.choice(range(i+1))) for i in range(font_number,0,-1)]
        for i in range(len(FONTS_SET),font_number, -1):
            gen_fonts.pop(random.choice(range(0,len(gen_fonts))))

    dictionary = _initialize_dictionary()
    generator = ImageCaptcha(fonts=gen_fonts)

    dataset = []

    for num in range(0,total):
        length_var = random.choice(range(4,9)) if length==0 else length
        label = ''
        for i in range(0, length_var):
            label += dictionary[random.choice(range(len(dictionary)))]

        generator.generate(label)
        generator.write(label, os.path.join(target_dir, label+'.png'))
        image = imgio.read_img_uc(os.path.join(target_dir, label+'.png'))
        dataset.append((image,label))

    return dataset

def download_captcha_dataset(website, total=1000):
    """
    TODO
    :param website:
    :param total:
    :return:
    """
    pass




