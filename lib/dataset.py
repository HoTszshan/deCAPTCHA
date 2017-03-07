"""
Get a dataset: Load, Generate, Download
"""
import os
import imgio as ImgIO
import random
from download import CaptchaSpider
from third_party.captcha.image import ImageCaptcha

FONTS_SET = ['/Library/Fonts/Arial.ttf', '/Library/Fonts/Brush Script.ttf', '/Library/Fonts/Phosphate.ttc']


def get_image_files(dir_path):
    return [os.path.join(dir_path,f) for f in os.listdir(dir_path)
            if f.endswith('.jpg')
            or f.endswith('.png')]

def get_save_image_name(folder, label, img_type='png'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    file_list = list(filter(lambda f: f.split('-')[0].lower()==label.lower(), os.listdir(folder)))
    filename = os.path.join(folder, label+'-'+str(len(file_list))+ '.' + img_type)
    return filename


# The file name of the captcha is its responding label
def __get_image_label_by_filename(image_path):
    image = ImgIO.read_image_uc(image_path)
    file_name = os.path.basename(image_path).split('.')[0]
    label = file_name.split('-')[0] if '-' in file_name else file_name
    #label = re.findall(r'^([0-9]+)-[0-9]+\..*$', file_name)[0]
    return (image,label)

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
def load_captcha_dataset(base_dir):
    files = get_image_files(base_dir)
    dataset =  map(__get_image_label_by_filename, files)
    return zip(*dataset)


def generate_captcha_dataset(target_dir='generate data', total=1000, length=4, font_number=1):
    # Make a dir if not exists
    folder = os.path.join('data', target_dir)
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

    #for num in range(0,total):
    def generate_captcha(num):
        length_var = random.choice(range(4,9)) if length==0 else length
        label = ''
        for i in range(0, length_var):
            label += dictionary[random.choice(range(len(dictionary)))]

        generator.generate(label)
        image_name = get_save_image_name(folder, label)
        generator.write(label, image_name)
        image = ImgIO.read_image_uc(image_name)
        dataset.append((image,label))

    map(generate_captcha, range(total))

    return dataset


def download_captcha_images(URL, total=1000):
    spider = CaptchaSpider(captchaURL=URL)
    spider.download_images(number=total)


# TODO: Persistence
#



