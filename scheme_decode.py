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
    decoder.plot_predict_process(image)
    return decoder.predict([image])[0]

def ndataset_decode(image):
    decoder = Profiler(os.path.join('model', 'ndataset.pkl'))#'dataset2.pkl'))#'ndataset.pkl'))
    decoder.plot_predict_process(image)
    return decoder.predict([image])[0]


def hkbu_decode(image):
    decoder = Profiler(os.path.join('model', 'hkbu1.pkl'))
    decoder.plot_predict_process(image)
    return decoder.predict([image])[0]

def hongxiu_decode(image):
    decoder = Profiler(os.path.join('model', 'hongxiu.pkl'))
    decoder.plot_predict_process(image)
    return decoder.predict([image])[0]

def wordpress_decode(image):
    decoder = Profiler(os.path.join('model', 'wordpress.pkl'))
    decoder.plot_predict_process(image)
    return decoder.predict([image])[0]

def xxsy_decode(image):
    decoder = Profiler(os.path.join('model', 'xxsy.pkl'))
    y =  decoder.predict([image])[0]
    result_list = []
    result_list.extend(y)
    for i in range(len(result_list)):
        if result_list[i] == '%':
            result_list[i] = '?'
    decoder.plot_predict_process(image)
    return ''.join(result_list)

def hku_decode(image):
    decoder = Profiler(os.path.join('model', 'samples.pkl'))
    decoder.plot_predict_process(image)
    return decoder.predict([image])[0]

def decode(image):
    if image.shape == (20, 60, 3):
        result = gdgs_decode(image)
        print "It belongs to gdgs scheme, the answer is %s" % result
        return result
    elif image.shape == (45, 140, 3):
        result = hkbu_decode(image)
        print "It belongs to hkbu scheme, the answer is %s" % result
        return result
    elif image.shape == (22, 63, 3):
        result = hongxiu_decode(image)
        print "It belongs to hongxiu scheme, the answer is %s" % result
        return result
    elif image.shape == (60, 175, 3):
        result = wordpress_decode(image)
        print "It belongs to wordpress scheme, the answer is %s" % result
        return result
    elif image.shape == (55, 175, 3):
        result = ndataset_decode(image)
        print "It belongs to dataset 2, the answer is %s" % result
        return result
    elif image.shape == (50, 200, 4):
        result = hku_decode(image)
        print "It belongs to dataset 2, the answer is %s" % result
        return result
    else:
        result = xxsy_decode(image)
        print "It belongs to xxsy scheme, the answer is %s" % result
        # print image.shape, "not define!"
        return  result


# image = ImgIO.read_image(os.path.join('data', 'dataset', 'hongxiu', 'testing_set','0l688-2.png'))
# ImgIO.show_image(image)
# print ndataset_decode(image)
# decode(image)

def main():
    image = ImgIO.read_image(sys.argv[1])
    decode(image)

if __name__ == '__main__':
    main()

# X, y = dataset.load_captcha_dataset(os.path.join('data', 'dataset', 'dataset_2', 'testing'))
# result = [(decode(x) ==i) for x,i in zip(X, y)]
# print sum(result)
# print sum(result) / float(len(y))