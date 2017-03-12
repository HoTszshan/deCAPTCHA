"""
Get a dataset: Load, Generate, Download
"""
import re
import os
import imgio as ImgIO
import numpy as np
import random
from sklearn.externals import joblib
from download import CaptchaSpider
from third_party.captcha.image import ImageCaptcha

FONTS_SET = ['/Library/Fonts/Arial.ttf', '/Library/Fonts/Brush Script.ttf', '/Library/Fonts/Phosphate.ttc']


def get_image_files(dir_path):
    return [os.path.join(dir_path,f) for f in os.listdir(dir_path)
            if f.endswith('.jpg')
            or f.endswith('.png')]

def get_save_image_name(folder, label, img_type='png'):
    if not os.path.exists(folder):
        os.mkdir(folder)
    file_list = list(filter(lambda f: f.split('-')[0].lower()==label.lower(), os.listdir(folder)))
    filename = os.path.join(folder, label+'-'+str(len(file_list))+ '.' + img_type)
    return filename


# The file name of the captcha is its responding label
def __get_image_label_by_filename(image_path, split='-'):
    image = ImgIO.read_image(image_path)
    file_name = os.path.basename(image_path).split('.')[0]
    label = file_name.split(split)[0] if split in file_name else file_name
    #label = re.findall(r'^([0-9]+)-[0-9]+\..*$', file_name)[0]
    return (image,label)

def __get_image_label_by_file(image_path):
    image = ImgIO.read_image(image_path)
    file_name = image_path.split('.jpg')[0]
    if os.path.isfile(file_name):
        label = open(file_name, "r").read()[:-1]
        return (image, label)
    else:
        return


def __initialize_digit_dictionary():
    dictionary = []
    for index in range(48,58):
        dictionary.append(chr(index))
    return dictionary

def __initialize_dictionary():
    dictionary = []
    for index in range(48,58):
        dictionary.append(chr(index))
    for index in range(65,91):
        dictionary.append(chr(index))
    for index in range(97, 123):
        dictionary.append(chr(index))
    return dictionary



# Load captcha dataset depend on the image fold path
def load_captcha_dataset(base_dir, label_file=None, other_file=False, save_pkl=True, split_symbol='-'):
    other_file = True if label_file and not other_file else other_file
    files = get_image_files(base_dir)#get_image_files(os.path.join(FOLDER, base_dir))
    if not other_file:
        dataset =  map(lambda x:__get_image_label_by_filename(x,split_symbol), files)
    elif label_file:
        #tmp_dataset = map(lambda f: (ImgIO.read_image(f), os.path.basename(f).split('.')[0]), files)
        tmp_dataset = {}
        dataset = []
        for f in files: tmp_dataset[os.path.basename(f).split('.')[0]] = ImgIO.read_image(f)
        file_path = os.path.join(base_dir,label_file)
        pattern_line = re.compile(r'(\d+) (\w+) (\w+)')
        with open(file_path, "r") as txt_file:
            result = txt_file.readlines()
            for line in result:
                tmp = re.findall(pattern_line, line)[0]
                index = tmp[0]
                label = tmp[1] + ' ' + tmp[2]
                dataset.append((tmp_dataset[index], label))
    else:
        dataset =  filter(lambda x: x, map(__get_image_label_by_file, files))

    if save_pkl:
        __dump_dataset_pkl(zip(*dataset), base_dir)
    return zip(*dataset)


def generate_captcha_dataset(target_dir='generate data', n_samples=1000, length=4, font_number=1, save_pkl = True):
    # Make a dir if not exists
    folder = target_dir#os.path.join(FOLDER, target_dir)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Set the font set
    gen_fonts = FONTS_SET[:]
    if 0 < font_number < len(FONTS_SET):
        #fonts = [fonts.pop(random.choice(range(i+1))) for i in range(font_number,0,-1)]
        for i in range(len(FONTS_SET),font_number, -1):
            gen_fonts.pop(random.choice(range(0,len(gen_fonts))))
    dictionary = __initialize_dictionary()
    generator = ImageCaptcha(fonts=gen_fonts)

    dataset = []
    #for num in range(0,n_samples):
    def generate_captcha():
        length_var = random.choice(range(4,9)) if length==0 else length
        label = ''
        for i in range(0, length_var):
            label += dictionary[random.choice(range(len(dictionary)))]
        generator.generate(label)
        image_name = get_save_image_name(folder, label)
        generator.write(label, image_name)
        image = ImgIO.read_image(image_name)
        dataset.append((image,label))
    map(lambda x: generate_captcha(), range(n_samples))
    if save_pkl:
        __dump_dataset_pkl(zip(*dataset), target_dir)
    return zip(*dataset)


def download_captcha_images(URL, n_samples=1000):
    spider = CaptchaSpider(captchaURL=URL)
    spider.download_images(number=n_samples)

def __dump_dataset_pkl(data, filename):
    pkl_name = filename  if len(filename.split('.pkl')) >=2 else filename + '.pkl'
    joblib.dump(data, pkl_name)#os.path.join(FOLDER, pkl_name))

def load_captcha_pkl(filename):
    pkl_name = filename  if len(filename.split('.pkl')) >=2 else filename + '.pkl'
    return joblib.load(pkl_name)#os.path.join(FOLDER, pkl_name))

def get_single_label_unique(label_list):
    labels = []
    labels.extend(reduce((lambda x, y: x+y), label_list))
    return list(set(labels))

def stratified_shuffle_split(images, labels, test_size=0.3, save_dir=None):
    dataset = zip(images, labels)
    length = len(labels[0])
    keys = get_single_label_unique(labels)
    dict_keys = {}
    for index, key in enumerate(keys):
        dict_keys[key] = index
    threshold = int(len(dataset) * length * test_size / float(len(keys)))

    threshold_array = np.full(len(keys), threshold, dtype=np.int8)
    flag_array = np.full(len(keys), 0, dtype=np.int8)
    mat_stat, index_list = [], []
    while not np.all(flag_array==threshold_array):
        random.shuffle(dataset)
        mat_stat = np.zeros((length+1, len(keys)))

        index_list=[]
        for i, (image, label) in enumerate(dataset):
            tmp_add = mat_stat[length, :].copy()
            for l in label: tmp_add[dict_keys[l]] += 1
            if np.any(tmp_add > threshold_array):
                continue
            for j in range(len(label)):
                mat_stat[j, dict_keys[label[j]]] += 1
                mat_stat[length, dict_keys[label[j]]] += 1
            index_list.append(i)
        flag_array = mat_stat[-1, :].copy()
        if flag_array.std() ** 2 < 0.5: break

    testing_set = []
    def split(i):
        if i in index_list:
            testing_set.append(dataset[i])
        else:
            training_set.append(dataset[i])

    map(split, range(len(dataset)))
    training_images, training_labels = zip(*training_set)
    testing_images, testing_labels = zip(*testing_set)
    if save_dir:
        folder = save_dir#os.path.join(FOLDER,save_dir)
        if not os.path.exists(folder):
            os.mkdir(folder)
        training_folder = os.path.join(folder, 'training_set')
        testing_folder =  os.path.join(folder, 'testing_set')
        map(lambda image,label:ImgIO.write_image(image,get_save_image_name(training_folder,label)), training_images, training_labels)
        map(lambda image,label:ImgIO.write_image(image,get_save_image_name(testing_folder,label)), testing_images, testing_labels)
    return training_images, training_labels, testing_images, testing_labels


