# extract simple and basic features
from feature import process2
from lib import imgio as ImgIO

class ScaleExtract(object):
    def __init__(self, extract_func):
        self.callback = extract_func

    def __scale_down(self, image):
        #ImgIO.show_image(process2.resize_transform(image, size=(16,16)))
        return process2.resize_transform(image, size=(16,16))

    def __call__(self, image, features):
        return self.callback(self.__scale_down(image), features, prefix='scaled-')


# Extract function

def position_brightness(image, features, prefix=''):
    height, width = image.shape[0], image.shape[1]
    for y in range(width):
        for x in range(height):
            features[prefix+'pos-'+str(x*height+y)] = image[x, y]