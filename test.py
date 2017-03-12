import lib.imgio as ImgIO
import os
import numpy as np
from lib import dataset
import re
import time
from lib.util import *
from feature.shapecontext import *
from feature import process2
from feature.simple import *
from model.individual import *
from lib.segment import *
from multiprocessing.dummy import Pool as ThreadPool
from lib import util
import sys
sys.setrecursionlimit(10000)

def pre_processing_digit(image):
    img = process.filter_inverse(image)
    img = process2.rgb_to_gray(img)
    img = process.filter_reduce_lines(img, median=200)
    img = process.filter_mean_smooth(img)
    img = process.filter_threshold_RGB(img, threshold=150)
    img = process.filter_fix_broken_characters(img)
    img = process.filter_fill_holes(img)
    img = process.filter_fix_broken_characters(img)
    #img = process.filter_erosion(img, window_size=2)
    #img = process.filter_dilation(img, window_size=3)
    #ImgIO.show_image(img)
    return img

def estimate_function_time(function, n_iter=1, input_list=None, **params):
    start_time = time.time()
    if input_list:
        if params:
            def func(x):
                return function(x, **params)
            result = map(lambda n:map(func, input_list), range(n_iter))
        else:
            result = map(lambda n:map(function, input_list), range(n_iter))
    else:
        result = map(function, range(n_iter))
    finish_time = time.time()
    print ("It takes %.4f s to test %s function %d times." % ((finish_time - start_time), function.__name__, n_iter))
    return filter(lambda x:not x == None, result)[0] if len(filter(lambda x:not x == None, result)) > 0 else None

def copy_images(filename, target_folder):
    image = ImgIO.read_image(filename)
    target = dataset.get_save_image_name(target_folder, os.path.basename(filename).split('.')[0], img_type='jpg')
    ImgIO.write_image(image, target)


def test_char(filename):
    i = ImgIO.read_image(filename)
    processor = ComposeProcessor(processors=[#(process2.inverse, None),
        (process2.smooth, {'sigma':3}),
        (process2.sci_median, {'size': 3}),
        (process2.otsu_filter, None),#(process2.erosion, None),
        #(process2.sci_median, {'size': 3}),
        #(process2.dilation, None),
        #(process2.otsu_filter, None),
        #(process2.sci_median, {'size': 3}),
        #(process.filter_fix_broken_characters, None),
        #(process.filter_fill_holes,None),
        #(process2.denosie_color_filling, {'size': 32}),
        (process2.extract_skeleton, None)
    ])
    j = processor(i)#, show_process=True)
    #ImageIO.show_image(new_image)
    #k = #process2.extract_skeleton(
    k = process.filter_average_image(map(ImgIO.read_image_uc, dataset.get_image_files('test_easy_digits/1')))
    items = filename.split('/')[:-1]
    new_folder = 'data'
    for i in items:
        new_folder = os.path.join(new_folder, i)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
    #ImgIO.show_images_list([i, j, processor(k)])
    #print j.sum()
    ImgIO.write_image(j, os.path.join(new_folder, os.path.basename(filename)))
    return j.sum()#'%4d' % j.sum(), '%4d' % processor(k).sum(), i.shape
    # img = process.filter_inverse(image)
    # img = process.filter_reduce_lines(img, median=200)
    # img = process.filter_mean_smooth(img)
    # img = process.filter_threshold_RGB(img, threshold=150)
    # img = process.filter_fix_broken_characters(img)
    # img = process.filter_fill_holes(img)
    # img = process.filter_fix_broken_characters(img)


def test_rename_g():
    test_digit_folder = 'test_easy_digits'
    folder_list = [os.path.join(test_digit_folder,f) for f in os.listdir(test_digit_folder)
                   if os.path.isdir(os.path.join(test_digit_folder,f)) and len(f) < 2]


    for f in folder_list:
        print os.path.basename(f),
        # for s in dataset.get_image_files(f): print s
        t = estimate_function_time(test_char, input_list=dataset.get_image_files(f))
        print '\t', min(t), [v for v in t if v <= 80]
        #print "Min: " % min(t), t


def test():
    file_name_3 = 'test_funcs/test_easy_digits/5/5.jpg'
    file_name_4 = 'test_funcs/test_easy_digits/6/11.jpg'
    file_name_5 = 'test_funcs/test_easy_digits/6/33.jpg'
    sc1 = ShapeContext(ImgIO.read_image_uc(file_name_3), sample_num=85)
    sc2 = ShapeContext(ImgIO.read_image_uc(file_name_4), sample_num=85)
    sc3 = ShapeContext(ImgIO.read_image_uc(file_name_5), sample_num=85)
    #ImgIO.show_images_list([ImgIO.read_image_uc(file_name_3), ImgIO.read_image_uc(file_name_4)])
    scm1 = ShapeContextMatcher(sc1, sc2)
    scm1.display_tps_wrapping_result()
    scm2 = ShapeContextMatcher(sc2, sc3)
    scm2.display_tps_wrapping_result()
    print scm1.calculate_shape_distance(), scm2.calculate_shape_distance()

#estimate_function_time(test2,input_list=dataset.get_image_files('data/samples')[:1])

def test_sc_value(i):
    folder = 'test_funcs/test_easy_digits'
    folders = [os.path.join(folder, f) for f in os.listdir(folder) if len(f) <= 1]
    sc_dict = {}
    for f in folders:
        sc_dict[os.path.basename(f)] = map(lambda x: ShapeContext(ImgIO.read_image_uc(x), sample_num=85), dataset.get_image_files(f)[::5])

    def cal_s_l(n1, n2, sc, sc_list, verbose=False):
        dis = map(lambda x: ShapeContextMatcher(sc, x).calculate_shape_distance(), sc_list)
        if verbose:
            print "Shape distance between %s and %s : (Max: %.6f \t Min: %.6f \t Mean: %.6f)" % \
              (n1, n2, max(dis), min(dis), sum(dis) / len(dis))
        return (max(dis), min(dis), sum(dis) / len(dis))

    def cal_s_all(n, sc):
        dis = map(lambda x: (x[0], cal_s_l(n, x[0], sc, x[1])), sc_dict.items())
        t1 = sorted(dis, lambda x, y: cmp(x[1][1], y[1][1]))
        print "##### Min Shape distance between %s and %s :  Mean: %.6f)" % (n, t1[0][0], t1[0][1][1])
        t2 = sorted(dis, lambda x, y: cmp(x[1][2], y[1][2]))
        print "##### Mean Shape distance between %s and %s :  Mean: %.6f)" % (n, t2[0][0], t2[0][1][2])
        return t1[0][0], t1[0][1][1]

    def cal_s_s(n):
        dis = map(lambda x: cal_s_l(n, n, x, sc_dict[n], verbose=False), sc_dict[n])
        dis1, dis2, dis3 = zip(*dis)
        d = []
        d.extend(dis1)
        d.extend(dis2)
        d.extend(dis3)
        print "@@@@@@ Shape distance in %s: (Max: %.6f \t Min: %.6f \t Mean: %.6f)" % \
            (n, max(dis1), min(dis2), sum(d) / len(d))

    def cal_l_l(n):
        dis = map(lambda sc: cal_s_all(n, sc), sc_dict[n])
        t1 = sorted(dis, lambda x, y: cmp(x[1], y[1]))
        print "@@@@@ Min Shape distance is %s and %s : %.6f" % (n, t1[0][0], t1[0][1])
        return t1[0][0], t1[0][1]

    #cal_s_all('0', sc_dict['0'][0])
    #map(lambda x: cal_s_s(x), sc_dict.keys())
    map(lambda x: cal_l_l(x), sc_dict.keys())

def test_processing_image(image):
    #"""
    processor = ComposeProcessor(processors=[(process2.inverse, None),
                                             #(process2.sci_median, {'size': 3}),
                                             (process2.rgb_to_gray, None),
                                             (process.filter_reduce_lines, {'median': 200}),
                                             (process2.threshold_filter, {'threshold':150}),#(process2.otsu_filter, None),
                                             (process2.sci_median, {'size': 3}),
                                             (process.filter_fix_broken_characters, None),
                                             (process.filter_fill_holes,None),
                                             (process.filter_fix_broken_characters, None),
                                             #(process2.reconstruction, None),
                                             #(process2.denosie_color_filling, {'size': 32}),
                                             #(process2.extract_skeleton, None)
                                             (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
                                             ])
    #"""
    #processor = pre_processing_digit
    new_image = processor(image)
    #ImgIO.show_images_list(CharacterSeparator(new_image).get_characters())
    return new_image

#estimate_function_time(test_sc_value)

def test_processing(folder):
    # It takes 5.1154 s to test test_processing function 1 times.
    # For: It takes 23.6704 s to test test_processing function 1 times.
    # Map: It takes 22.9716 s to test test_processing function 1 times.
    # Multithread : It takes 29.2302 s to test test_processing function 1 times.
    """
    processor = ComposeProcessor(processors=[(process2.inverse, None),
                                             #(process2.sci_median, {'size': 3}),
                                             (process2.rgb_to_gray, None),
                                             (process.filter_reduce_lines, {'median': 200}),
                                             (process2.threshold_filter, {'threshold':150}),#(process2.otsu_filter, None),
                                             (process2.sci_median, {'size': 3}),
                                             (process.filter_fix_broken_characters, None),
                                             (process.filter_fill_holes,None),
                                             (process.filter_fix_broken_characters, None),
                                             #(process2.reconstruction, None),
                                             #(process2.denosie_color_filling, {'size': 32}),
                                             #(process2.extract_skeleton, None)
                                             (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
                                             ])
    #"""
    processor = pre_processing_digit
    #new_image = processor(image, show_process=True)
    images, labels = dataset.load_captcha_dataset(folder, save_pkl=False)

    image_list = map(lambda x:processor(x), images)
    #image_list = [processor(i) for i in images]

    #ImgIO.show_images_list(image_list)
    #print "process"
    """
    # write
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    names = [os.path.join(target_folder, i +'.jpg') for i in labels]
    #ImgIO.print_image_array(image_list[0])
    #map(ImgIO.write_image, images, names)
    #map(ImgIO.show_image, image_list)
    #ImgIO.show_images_list(image_list[:10])
    """

def test_segment(folder):
    processor = ComposeProcessor(processors=[(process2.inverse, None),
                                             #(process2.sci_median, {'size': 3}),
                                             (process2.rgb_to_gray, None),
                                             (process.filter_reduce_lines, {'median': 200}),
                                             (process2.threshold_filter, {'threshold':150}),#(process2.otsu_filter, None),
                                             (process2.sci_median, {'size': 3}),
                                             (process.filter_fix_broken_characters, None),
                                             (process.filter_fill_holes,None),
                                             (process.filter_fix_broken_characters, None),
                                             #(process2.reconstruction, None),
                                             (process2.denosie_color_filling, {'size': 32}),
                                             #(process2.extract_skeleton, None)
                                             (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
                                             ])
    images, labels = dataset.load_captcha_dataset(folder)
    image_list = map(lambda x:processor(x), images[:10])
    separator = CharacterSeparator
    for image in image_list:
        imgs = separator(image).get_characters()
        #separator(image).show_split_chunks()
        ImgIO.show_images_list(imgs)

def test_extract_features(folder):
    processor = ComposeProcessor(processors=[(process2.inverse, None),
                                             #(process2.sci_median, {'size': 3}),
                                             (process2.rgb_to_gray, None),
                                             (process.filter_reduce_lines, {'median': 200}),
                                             (process2.threshold_filter, {'threshold':150}),#(process2.otsu_filter, None),
                                             (process2.sci_median, {'size': 3}),
                                             (process.filter_fix_broken_characters, None),
                                             (process.filter_fill_holes,None),
                                             (process.filter_fix_broken_characters, None),
                                             #(process2.reconstruction, None),
                                             (process2.denosie_color_filling, {'size': 32}),
                                             #(process2.extract_skeleton, None)
                                             (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
                                             ])
    images, labels = dataset.load_captcha_dataset(folder)
    image_list = map(lambda x:processor(x), images[:10])
    separator = CharacterSeparator
    imgs = []
    map(lambda x: imgs.extend(separator(x).get_characters()), image_list)
    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    dict_list = map(extractor, imgs)
    print len(dict_list), len(dict_list[0])
    for i in dict_list[0].items():
        print i
    #ImgIO.show_images_list(imgs)

target_folder = 'data/new'
folder = 'data/annotated_captchas/dataset'#'data/gimpy-r-ball'
test_folder = 'data/annotated_captchas/train'
train_folder = 'data/annotated_captchas/test'
#test_processing(folder)



def test_svm(train, test):
    sys.setrecursionlimit(10000)
    #"""
    processor = ComposeProcessor(processors=[(process2.inverse, None),
                                             #(process2.sci_median, {'size': 3}),
                                             (process2.rgb_to_gray, None),
                                             (process.filter_reduce_lines, {'median': 200}),
                                             (process2.threshold_filter, {'threshold':150}),#(process2.otsu_filter, None),
                                             (process2.sci_median, {'size': 3}),
                                             (process.filter_fix_broken_characters, None),
                                             (process.filter_fill_holes,None),
                                             (process.filter_fix_broken_characters, None),
                                             #(process2.reconstruction, None),
                                             #(process2.denosie_color_filling, {'size': 32}),
                                             #(process2.extract_skeleton, None)
                                             (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
                                             ])
    #"""
    #processor = pre_processing_digit
    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = SVMEngine()
    decoder = CaptchaDecoder(processor=processor, separator=CharacterSeparator, extractor=extractor, engine=engine)
    images, labels = dataset.load_captcha_dataset(train, save_pkl=False)
    print ("Number of training set: %d" % len(images))
    decoder.fit(images, labels)

    test_i, test_l = dataset.load_captcha_dataset(test, save_pkl=False)
    print ("Number of testing set: %d" % len(test_i))
    print "Score:", decoder.score(test_i, test_l, verbose=True)


def test_framework(folder):
    print("Testing framework...\n")

    # print("Shuffle split:")
    # images, labels = dataset.load_captcha_dataset(folder, save_pkl=True)
    # training_images, training_labels, testing_images, testing_labels = dataset.stratified_shuffle_split(images, labels,
    #                                                                                                     save_dir=folder)
    # training_images, training_labels = dataset.load_captcha_dataset(os.path.join(folder, 'training_set'), save_pkl=True)
    # testing_images, testing_labels = dataset.load_captcha_dataset(os.path.join(folder, 'testing_set'), save_pkl=True)
    # print("Number of training set: %d" % len(training_images))
    # print("Number of testing set: %d" % len(testing_images))

    training_images, training_labels = dataset.load_captcha_pkl(os.path.join(folder, 'training_set.pkl'))
    testing_images, testing_labels = dataset.load_captcha_pkl(os.path.join(folder, 'testing_set'))

    print('Training:')
    processor = ComposeProcessor(processors=[(process2.inverse, None),
                                             #(process2.sci_median, {'size': 3}),
                                             (process2.rgb_to_gray, None),
                                             (process.filter_reduce_lines, {'median': 200}),
                                             (process2.threshold_filter, {'threshold':150}),#(process2.otsu_filter, None),
                                             (process2.sci_median, {'size': 3}),
                                             (process.filter_fix_broken_characters, None),
                                             (process.filter_fill_holes,None),
                                             (process.filter_fix_broken_characters, None),
                                             #(process2.reconstruction, None),
                                             #(process2.denosie_color_filling, {'size': 32}),
                                             #(process2.extract_skeleton, None)
                                             (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
                                             ])
    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = SVMEngine()
    decoder = CaptchaDecoder(processor=processor, separator=CharacterSeparator, extractor=extractor, engine=engine)
    decoder.fit(training_images, training_labels)
    util.save_model(decoder, os.path.join(folder, 'model.pkl'))
    print('')
    print('Testing:')
    print "Score:", decoder.score(testing_images, testing_labels)

def test_model(folder):
    t1 = time.time()
    test = os.path.join(folder, 'testing_set')
    testing_images, testing_labels = dataset.load_captcha_dataset(test, save_pkl=False)
    #estimate_function_time(Profiler, input_list=[os.path.join(folder, os.path.basename(folder)+'.pkl')])
    profiler = Profiler(os.path.join(folder, os.path.basename(folder)+'.pkl'))
    print("It takes %.4f s to load model." % (time.time() - t1))
    t2= time.time()
    profiler.print_score(testing_images, testing_labels, verbose=False)
    print("It takes %.4f s to get socres." % (time.time() - t2))
    print("Size of testing set is %d." % len(testing_images))

def test_grid_search(folder):
    train = os.path.join(folder, 'training_set')
    test = os.path.join(folder, 'testing_set')
    training_images, training_labels = dataset.load_captcha_dataset(train, save_pkl=False)
    testing_images, testing_labels = dataset.load_captcha_dataset(test, save_pkl=False)
    processor = ComposeProcessor(processors=[(process2.inverse, None),
                                             #(process2.sci_median, {'size': 3}),
                                             (process2.rgb_to_gray, None),
                                             (process.filter_reduce_lines, {'median': 200}),
                                             (process2.threshold_filter, {'threshold':150}),#(process2.otsu_filter, None),
                                             (process2.sci_median, {'size': 3}),
                                             (process.filter_fix_broken_characters, None),
                                             (process.filter_fill_holes,None),
                                             #(process.filter_fix_broken_characters, None),
                                             (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
                                             ])

    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = SVMEngine()
    decoder = CaptchaDecoder(processor=processor, separator=CharacterSeparator, extractor=extractor, engine=engine)
    decoder.fit(training_images, training_labels, grid_search = True)
    decoder.score(testing_images, testing_labels, verbose=True)

#test_processing(folder)
#test_segment(folder)
#test_extract_features(folder)
#estimate_function_time(test_extract_features, input_list=[folder])
#print "Train: ",estimate_function_time(test_processing, input_list=[folder], n_iter=1)
#test_extract_features(folder)

#print "test svm: \n", test_svm(train_folder, test_folder)
# print len(dataset.get_image_files(os.path.join(folder,'training_set')))
# print len(dataset.get_image_files(os.path.join(folder,'testing_set')))
#print "test_model:\n", test_model(folder)
#
# print os.path.join(folder,'training_set')
# print os.path.join(folder,'testing_set')
# estimate_function_time(test_framework, input_list=[folder])
#estimate_function_time(test_processing_image, input_list=map(ImgIO.read_image,dataset.get_image_files(folder)), n_iter=1)
#print "test grid search: \n", test_grid_search(folder)


#test_rename_g()
#estimate_function_time(copy_images, input_list=dataset.get_image_files('lib/data/email'), target_folder='lib/data/163_1000')