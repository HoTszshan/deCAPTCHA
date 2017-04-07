from feature.shapecontext import *
from feature.simple import *
from lib import util
from lib.util import *
from model.individual import *
from segment.color import *
from sklearn.externals import joblib
sys.setrecursionlimit(10000)

def gdgs_decode(image):
    decoder = Profiler(os.path.join('model', 'gdgs.pkl'))
    return decoder.predict([image])[0]

def ndataset_decode(image):
    decoder = Profiler(os.path.join('model', 'ndataset.pkl'))
    return decoder.predict([image])[0]


def hkbu_decode(image):
    decoder = Profiler(os.path.join('model', 'hkbu.pkl'))
    return decoder.predict([image])[0]

def hongxiu_decode(image):
    decoder = Profiler(os.path.join('model', 'hongxiu.pkl'))
    return decoder.predict([image])[0]

def wordpress_decode(image):
    decoder = Profiler(os.path.join('model', 'wordpress.pkl'))
    return decoder.predict([image])[0]


def decode(image):
    if image.shape == (20, 60, 3):
        print "It belongs to gdgs scheme, the answer is %s" % gdgs_decode(image)
    elif image.shape == (45, 140, 3):
        print "It belongs to hkbu scheme, the answer is %s" % hkbu_decode(image)
    elif image.shape == (22, 63, 3):
        print "It belongs to hongxiu scheme, the answer is %s" % hongxiu_decode(image)
    elif image.shape == (60, 175, 3):
        print "It belongs to wordpress scheme, the answer is %s" % wordpress_decode(image)
    elif image.shape == (55, 175, 3):
        print "It belongs to ndataset scheme, the answer is %s" % ndataset_decode(image)
    else:
        print image.shape, "not define!"


# image = ImgIO.read_image(os.path.join('data', 'dataset', 'ndataset', 'testing_set','0108-0.png'))
# print ndataset_decode(image)
# decode(image)

def main():
    image = ImgIO.read_image(sys.argv[1])
    decode(image)

if __name__ == '__main__':
    main()