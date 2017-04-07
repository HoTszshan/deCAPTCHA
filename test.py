#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys

from feature.shapecontext import *
from feature.simple import *
from lib import util
from lib.util import *
from model.individual import *
from segment.color import *

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

def copy_image(filename, target_folder):
    image = ImgIO.read_image(filename)
    target = dataset.get_save_image_name(target_folder, os.path.basename(filename).split('-')[0], img_type='jpg')
    ImgIO.write_image(image, target)

def copy_images_from(folder, target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    filenames = dataset.get_image_files(folder)
    print(len(filenames))
    map(lambda x:copy_image(x, target_dir), filenames)
# copy_images_from('data/gregwar', 'data/gregwar_1')
# print len(dataset.get_image_files('data/gregwar'))
# l = dataset.get_image_files('data/gregwar')
# for i in l:
#     print i.split('-'), dataset.get_save_image_name('data/gregwar', os.path.basename(i).split('-')[0])

def test_char(filename):
    i = ImgIO.read_image(filename)
    processor = ComposeProcessor(processors=[
        #lambda x:(process2.resize_transform(x, output_shape=(100,50))),
        (process2.inverse),
        #(lambda x:process2.smooth(x, sigma=3)),
        #(process2.sci_median, {'size': 3}),
        #(process2.rgb_to_gray, {'r':0.3, 'g':0.59, 'b':0.11}),
        #(process2.rgb_to_gray),
        #(process2.otsu_filter),#(process2.erosion, None),
        #(lambda x:process2.threshold_filter(x, threshold=200)),
        #(process2.sci_median, {'size': 3}),
        process2.smooth,
        #(process2.dilation),
        #(process2.closing),
        (process2.otsu_filter),
        #(process2.sci_median, {'size': 3}),
        #(process.filter_fix_broken_characters, None),
        #(process.filter_fill_holes,None),
        #(process2.denosie_color_filling, {'size': 32}),
        (process2.extract_skeleton)
    ])
    j = processor(i)#, show_process=True)
    #ImgIO.show_images_list([i, j])
    #ImgIO.show_images_list([j,process2.normalise_scaling(j, output_shape=(120,180))], nearest=True)
    k = process2.normalise_rotation(j)
    #ImageIO.show_image(new_image)
    #k = #process2.extract_skeleton(
    # k = process.filter_average_image(map(ImgIO.read_image_uc, dataset.get_image_files('test_easy_digits/1')))
    # items = filename.split('/')[:-1]
    # new_folder = 'data'
    # for i in items:
    #     new_folder = os.path.join(new_folder, i)
    #     if not os.path.exists(new_folder):
    #         os.mkdir(new_folder)
    #ImgIO.show_images_list([i, j, processor(k)])
    #print j.sum()
    #ImgIO.write_image(j, os.path.join(new_folder, os.path.basename(filename)))
    #'%4d' % j.sum(), '%4d' % processor(k).sum(), i.shape
    ImgIO.show_images_list([i, j, k])


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

def test_processing_one_image(image):
    processor = ComposeProcessor(processors=[# (process2.inverse),
        # (process2.threshold_RGB_filter, {'threshold':(30,30,30)}),

        (process2.rgb_to_gray, None),
        (process2.otsu_filter, None),
        (process2.reconstruction, None),
        (process.filter_remove_dots, None),
        (process2.denosie_color_filling, None),

        (lambda img:process.filter_reduce_lines(img, median=0.78), None),
        # (process.filter_reduce_lines, None)
                                             ])
    #"""

    #processor = pre_processing_digit
    new_image = processor(image)
    ImgIO.show_images_list([image, new_image])
    sp = ColorFillingSeparator(new_image, length=4)
    sp.segment_process()
    return new_image

def test_cluster_segemt(image):
    s = ColorClusterSeparator(image)
    s.kmeans_segment()
    s.display_segment_result()
# test_cluster_segemt(ImgIO.read_image('data/xxsy/0+15=%_0.jpg'))

def test_segmentation_one_image(image):
    processor = ComposeProcessor(processors=[
        (process2.inverse, None),
        (process2.rgb_to_gray, None),
        (process.filter_reduce_lines, {'median':200}),
        (process.filter_mean_smooth, None),
        (process.filter_threshold_RGB,{'threshold':150}),
        # (process2.sci_median, {'size': 3}),
        (process.filter_fix_broken_characters, None),
        (process.filter_fill_holes,None),
        (process.filter_fix_broken_characters, None),
        (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
    ])
    tmp = processor(image)
    ImgIO.show_images_list([image,tmp])
    ImgIO.show_image(tmp)
    separator = ColorFillingSeparator(tmp)#SnakeSeparator(tmp)
    ImgIO.show_images_list(separator.get_characters())
    separator.save_segment_result('data', '0035')
    # separator.display_snake_segment()
    # separator.display_segment_result()
    # ImgIO.show_images_list(separator.get_characters())
# estimate_function_time(test_sc_value)
# test_segmentation_one_image(ImgIO.read_image_uc('data/annotated_captchas/dataset/0035-0.jpg'))


def test_processing(folder):
    # It takes 5.1154 s to test test_processing function 1 times.
    # For: It takes 23.6704 s to test test_processing function 1 times.
    # Map: It takes 22.9716 s to test test_processing function 1 times.
    # Multithread : It takes 29.2302 s to test test_processing function 1 times.
    processor = ComposeProcessor(processors=[# (process2.inverse),
        # (process2.threshold_filter, {'threshold':100}),
        # (process2.denosie_color_filling,{'remove_size':8, 'n_4':False}),
        # (process.filer_reduce_mesh, None),
        # # (process2.smooth, {'sigma':2}),
        # # (process2.sharpen, None),
        # # (process2.smooth, None),
        # (process2.threshold_filter, {'threshold':100}),
        # (process2.dilation, None),
        # (process.filer_reduce_mesh, None),
        # (process2.denosie_color_filling,{'remove_size':60, 'n_4':False}),
        (process2.rgb_to_gray, None),
        (process2.threshold_filter, {'threshold':45}),
        (process2.dilation,{'structure': np.ones((3,3))} ),
        (process.filter_fill_holes, None),
        (process.filter_fix_broken_characters, None)
    ])
    #"""
    #"""
    #processor = pre_processing_digit
    #new_image = processor(image, show_process=True)
    images, labels = dataset.load_captcha_dataset(folder, save_pkl=False)

    image_list = map(lambda x:processor(x), images[:36])
    #image_list = [processor(i) for i in images]
    for i in range(0, len(image_list), 6):
        upper = min(i+6, len(image_list))
        ImgIO.show_images_list(image_list[i:upper])
    #print "process"
    #"""
    print len(images)
    #os.mkdir(os.path.join('result', 'processing'))
    new_folder = os.path.join('result', 'processing', os.path.basename(folder))
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    map(lambda x,y: ImgIO.write_image(x, dataset.get_save_image_name(new_folder, y, img_type='jpg')), image_list, labels)
    #ImgIO.print_image_array(image_list[0])
    #map(ImgIO.write_image, images, names)
    #map(ImgIO.show_image, image_list)
    #ImgIO.show_images_list(image_list[:10])
    #"""
# test_processing_one_image(ImgIO.read_image('data/new_one/2Nsz-0.png'))
def test_segment(folder):
    processor = ComposeProcessor(processors=[
        (process2.inverse, None),
        (process2.rgb_to_gray, None),
        (process.filter_reduce_lines, {'median':200}),
        (process.filter_mean_smooth, None),
        (process.filter_threshold_RGB,{'threshold':150}),
        # (process2.sci_median, {'size': 3}),
        (process.filter_fix_broken_characters, None),
        (process.filter_fill_holes,None),
        (process.filter_fix_broken_characters, None),
        (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
    ])
    images, labels = dataset.load_captcha_dataset(folder)
    image_list = map(lambda x:processor(x), images)
    #ImgIO.show_images_list(image_list)
    new_folder = os.path.join('result', 'segment', os.path.basename(folder))
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    #ImgIO.show_images_list(image_list)
    # map(lambda x,y: ImgIO.write_image(x, dataset.get_save_image_name(new_folder, y,
    #                                                                  img_type='jpg')), image_list, labels[:10])
    separator = ColorFillingSeparator#Separator
    for image, label in zip(image_list, labels):
        s = separator(image)
        ImgIO.show_images_list(s.get_characters())
        imgs = s.get_characters()

        s.save_segment_result(new_folder, label)
        # s.display_segment_result()
        # s.save_segment_result(new_folder, label, save_char=True)
        #separator(image).show_split_chunks()
        ImgIO.show_images_list(imgs)
    # gdgs = 'data/gdgs'
    # ImgIO.show_images_list([ImgIO.read_image('result/segment/gdgs/0024-0.jpg'),
    #                         ImgIO.read_image('result/segment/gdgs/0024_segment_result-0.jpg')])
# test_segment('data/annotated_captchas/dataset')


def test_extract_features(folder):
    processor = ComposeProcessor(processors=[(process2.inverse),
                                             (process2.rgb_to_gray),
                                             (lambda x: process.filter_reduce_lines(x, median=200)),
                                             (lambda x: process2.threshold_filter(x, threshold=150)),
                                             (lambda x: process2.sci_median(x, size=3)),
                                             (process.filter_fix_broken_characters),
                                             (process.filter_fill_holes),
                                             # (process.filter_fix_broken_characters),
                                             # (lambda x: process2.denosie_color_filling(x, remove_size=32))
                                             (lambda x:process2.max_min_normalization(x, max_value=255., min_value=0.))
                                             ])
    images, labels = dataset.load_captcha_dataset(folder)
    image_list = map(lambda x:processor(x), images[:10])
    separator = ColorFillingSeparator
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
# test_processing('data/gimpy-r-ball')



def test_svm(train, test):
    sys.setrecursionlimit(10000)
    #"""
    processor = ComposeProcessor(processors=[(process2.inverse),
                                             (process2.rgb_to_gray),
                                             (lambda x: process.filter_reduce_lines(x, median=200)),
                                             (lambda x: process2.threshold_filter(x, threshold=150)),
                                             (lambda x: process2.sci_median(x, size=3)),
                                             (process.filter_fix_broken_characters),
                                             (process.filter_fill_holes),
                                             # (process.filter_fix_broken_characters),
                                             # (lambda x: process2.denosie_color_filling(x, remove_size=32))
                                             # (process2.extract_skeleton),
                                             (lambda x:process2.max_min_normalization(x, max_value=255., min_value=0.))
                                             ])

    #"""
    #processor = pre_processing_digit
    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = SVMEngine()#SOMEngine(kshape=(3,4), niter=400, learning_rate=0.05)#SVMEngine()
    decoder = CaptchaDecoder(processor=processor, separator=ColorFillingSeparator, extractor=extractor, engine=engine)
    images, labels = dataset.load_captcha_dataset(train, save_pkl=False)
    print ("Number of training set: %d" % len(images))
    decoder.fit(images, labels)

    test_i, test_l = dataset.load_captcha_dataset(test, save_pkl=False)
    print ("Number of testing set: %d" % len(test_i))
    print "Score:", decoder.score(test_i, test_l, verbose=True)


def test_cnn(folder):
    print("Testing framework...\n")

def test_framework(folder):
    print("Testing framework...\n")
    sys.setrecursionlimit(10000)
    # print("Shuffle split:")
    # images, labels = dataset.load_captcha_dataset(folder, save_pkl=True)
    # training_images, training_labels, testing_images, testing_labels = dataset.stratified_shuffle_split(images, labels,
    #                                                                                                     save_dir=folder)
    # training_images, training_labels = dataset.load_captcha_dataset(os.path.join(folder, 'training_set'), save_pkl=True)
    # testing_images, testing_labels = dataset.load_captcha_dataset(os.path.join(folder, 'testing_set'), save_pkl=True)
    # print("Number of training set: %d" % len(training_images))
    # print("Number of testing set: %d" % len(testing_images))
    #
    training_images, training_labels = dataset.load_captcha_pkl(os.path.join(folder, 'training_set.pkl'))

    print('Training:')
    # processor = ComposeProcessor(processors=[(process2.inverse),
    #                                          (process2.rgb_to_gray),
    #                                          (lambda x: process.filter_reduce_lines(x, median=200)),
    #                                          (lambda x: process2.threshold_filter(x, threshold=150)),
    #                                          (lambda x: process2.sci_median(x, size=3)),
    #                                          (process.filter_fix_broken_characters),
    #                                          (process.filter_fill_holes),
    #                                          # (process.filter_fix_broken_characters),
    #                                          # (lambda x: process2.denosie_color_filling(x, remove_size=32))
    #                                          # (process2.extract_skeleton),
    #                                          (lambda x:process2.max_min_normalization(x, max_value=255., min_value=0.))
    #                                          ])


    processor = ComposeProcessor(processors=[
        #lambda x:(process2.resize_transform(x, output_shape=(100,50))),
        (process2.inverse, None),
        (process2.rgb_to_gray, None),
        (process.filter_reduce_lines, {'median':200}),
        #(process2.otsu_filter),#(process2.erosion, None),
        (process2.threshold_filter,{'threshold':150}),
        (process2.sci_median, {'size': 3}),
        # process2.smooth,
        # #(process2.dilation),
        # #(process2.closing),
        # (process2.otsu_filter),
        #(process2.sci_median, {'size': 3}),
        (process.filter_fix_broken_characters, None),
        (process.filter_fill_holes,None),
        #(process2.denosie_color_filling, {'size': 32}),
        # (process2.extract_skeleton)
        (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
    ])

    print len(training_images)#, len(testing_images)
    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = SOMEngine(kshape=(3,4), niter=4000, learning_rate=0.05)#SVMEngine()
    decoder = CaptchaDecoder(processor=processor, separator=ColorFillingSeparator, extractor=extractor, engine=engine)
    decoder.fit(training_images, training_labels)
    util.save_model(decoder, os.path.join(folder, 'toy_svm_model.pkl'))
    testing_images, testing_labels = dataset.load_captcha_pkl(os.path.join(folder, 'testing_set'))
    print('')
    print('Testing:')
    print "Score:", decoder.score(testing_images, testing_labels, verbose=True)

    profiler = Profiler(os.path.join(folder, 'toy_svm_model.pkl'))
    print "start"
    t2= time.time()
    profiler.print_score(testing_images, testing_labels, verbose=True)
    print("It takes %.4f s to get socres." % (time.time() - t2))
    print("Size of testing set is %d." % len(testing_images))

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
    processor = ComposeProcessor(processors=[
        (process2.inverse, None),
        (process2.rgb_to_gray, None),
        (process.filter_reduce_lines, {'median':200}),
        (process.filter_mean_smooth, None),
        (process.filter_threshold_RGB,{'threshold':150}),
        (process.filter_fix_broken_characters, None),
        (process.filter_fill_holes,None),
        (process2.max_min_normalization, {'max_value':255., 'min_value':0.})
    ])
    separator=ColorFillingSeparator
    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = SVMEngine(C=10.0, gamma=0.01)
    decoder = CaptchaDecoder(processor=processor, separator=separator, extractor=extractor, engine=engine)
    decoder.fit(training_images, training_labels, grid_search = False)
    print decoder.score(testing_images, testing_labels, verbose=True)
    # decoder.save_engine(os.path.basename(folder)+'.pkl')
    save_model(decoder, os.path.join('model', os.path.basename(folder)+'.pkl'))

    p = Profiler(os.path.join('model', os.path.basename(folder)+'.pkl'))
    print p.predict(testing_images[:2])
# test_grid_search(os.path.join('data', 'dataset', 'ndataset'))

def test_som():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, random_state=0)

    # som = SOM((4, 3), 400, learning_rate=0.05)
    # t1 = time.time()
    # som.train(X_train)
    # t2 = time.time()
    # print("It takes %.4f s to train som network" % (t2 - t1))
    # feature_dict = {}
    # for label, mapped in zip(y_train[::2], som(X_train[::2])):
    #     feature_dict[str(mapped)] = label
    # print len(feature_dict.items())
    # # for key, item in feature_dict.items():
    # #     print key, item
    #
    # predicted = [feature_dict[str(i)] for i in som(X_test)]

    som = SOMEngine(kshape=(10,10), niter=8000, learning_rate=0.05)
    #estimate_function_time(som.fit, input_list=[(som, X_train, y_train)])
    t1 = time.time()
    som.fit(X_train, y_train)
    t2 = time.time()
    print("It takes %.4f s to train som network" % (t2 - t1))

    predicted = som.predict(X_test)
    expected = y_test
    match = map(lambda x,y:x==y, predicted, expected)
    print "Accuracy: %.4f" % (sum(match) / float(len(y_test)))


#"""
#test_processing(folder)
# test_segmentation_one_image(ImgIO.read_image('data/annotated_captchas/dataset/1808-0.jpg'))


# test_segment(folder)
#test_extract_features(folder)
#estimate_function_time(test_extract_features, input_list=[folder])
#print "Train: ",estimate_function_time(test_processing, input_list=[folder], n_iter=1)
#test_extract_features(folder)

# print "test svm: \n", test_svm(train_folder, test_folder)
# print len(dataset.get_image_files(os.path.join(folder,'training_set')))
# print len(dataset.get_image_files(os.path.join(folder,'testing_set')))
#print "test_model:\n", test_model(folder)
#
# print os.path.join(folder,'training_set')
# print os.path.join(folder,'testing_set')


# estimate_function_time(test_framework, input_list=[folder])
#estimate_function_time(test_processing_image, input_list=map(ImgIO.read_image,dataset.get_image_files(folder)), n_iter=1)
#print "test grid search: \n", test_grid_search(folder)
# processing_folder = 'data/ez-gimpy'
# estimate_function_time(test_processing, input_list=[processing_folder])
# test_processing_one_image(ImgIO.read_image('data/gimpy-r-ball/0.1366121897745.jpg'))
#test_rename_g()
#estimate_function_time(copy_images, input_list=dataset.get_image_files('lib/data/email'), target_folder='lib/data/163_1000')


#test_char('test_easy_digits/1/11.jpg')
          #'data/annotated_captchas/dataset/2209-0.jpg')
# c_folder = 'test_easy_digits/1'
# estimate_function_time(test_char, input_list=dataset.get_image_files(c_folder))
# test_som()


# ImgIO.write_image(test_processing_one_image(ImgIO.read_image('data/test.jpg')), 'data/seg.jpg')
# test_segmentation_one_image(ImgIO.read_image('data/seg.jpg'))—
#estimate_function_time(test_segmentation_one_image, input_list=[ImgIO.read_image('data/seg.jpg')])
# gdgs = 'data/gdgs'
# estimate_function_time(test_segment, input_list=[gdgs])
#"""
# test_segment('data/annotated_captchas/dataset/training_set')

def test_winer(filename):
    from scipy import signal

    img = ImgIO.read_image_uc(filename)
    psf = np.ones((3, 3)) / 5.0
    # s_img1 = process.filter_mean_smooth(img, window_size=9)
    # s_img2 = process2.smooth(img)
    v_1 = process2.threshold_filter(img, 150)
    #restoration.wiener(img, psf, 100)#signal.wiener(s_img1)
    d_1 = process2.inverse(v_1)
    t_1 = process2.erosion(d_1)
    v_2 = signal.wiener(img)
    d_2 = process2.opening(process2.otsu_filter(v_2))
    t_2 = process2.inverse(d_2)
    ImgIO.show_images_list([img, v_1, d_1,t_1, v_2, d_2, t_2])


def split_train_test(folder, split='-'):
    folder_name = os.path.join('data', folder)
    images, labels = dataset.load_captcha_dataset(folder_name, save_pkl=False, split_symbol=split)
    target_folder = os.path.join('data', 'dataset', os.path.basename(folder_name))
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    dataset.stratified_shuffle_split(images, labels, test_size=0.3, save_dir=target_folder)
# split_train_test(os.path.join('annotated_captchas', 'ndataset'))

def k_train_test_split(folder, test_size=0.3):
    file_list = dataset.get_image_files(os.path.join('data', folder))
    random.shuffle(file_list)
    number = len(file_list) * test_size
    testing_file_list = []
    while len(testing_file_list) < number:
        choice = random.choice(file_list)
        if not choice in testing_file_list:
            testing_file_list.append(choice)
    target_folder = os.path.join('data', 'dataset', os.path.basename(folder))
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    training_folder = os.path.join(target_folder, 'training_set')
    testing_folder = os.path.join(target_folder, 'testing_set')
    os.mkdir(training_folder)
    os.mkdir(testing_folder)
    for f in file_list:
        image = ImgIO.read_image(f)
        if f in testing_file_list:
            ImgIO.write_image(image, os.path.join(testing_folder, os.path.basename(f)))
        else:
            ImgIO.write_image(image, os.path.join(training_folder, os.path.basename(f)))
# k_train_test_split('xxsy', test_size=0.3)
# split_train_test('hkbu_1000')
# estimate_function_time(split_train_test, input_list=['hongxiu', 'xxsy'])
# img2 = filters.wiener(img1)
# print type(img2)
# ImgIO.show_images_list([img1, img2])
# f = 'data/hkbu_1000/2d75-0.jpg'#'data/ez-gimpy/adjust.jpg'
# f = 'data/misc/' #'data/pconline_1000/'

# image = ImgIO.read_image('data/misc/92RY-0.tiff')
# image = ImgIO.read_image('data/hkgolden/8.jpg')
# print np.unique(process2.rgb_to_gray(image))
# ImgIO.show_images_list([image, process2.rgb_to_gray(image), process2.otsu_filter(process2.rgb_to_gray(image)),
#                         (process2.smooth(image)),
#                         process2.median(process2.otsu_filter(process2.rgb_to_gray(image)))])
# tmp = process2.median(image)
# ImgIO.show_images_list([image, tmp, process2.otsu_filter(tmp)])
def test_c_segment(filename):
    # image = ImgIO.read_image('data/WordPress/Confirmed__/22TF.png') #23HU.png
    image = ImgIO.read_image(filename)#'data/WordPress/Confirmed__/22My.png')
    separator = ColorClusterSeparator(image, n_cluster=12)
    separator.segment_process()
    # separator.display_kmeans_result()
    # separator.display_segment_result()
    print os.path.basename(filename)
    imgs = separator.get_characters()
    if not len(imgs) == 4:
        imgs.append(image)
        ImgIO.show_images_list(imgs)
# test_c_segment('data/word-press/15.jpg')
# estimate_function_time(test_c_segment, input_list=dataset.get_image_files('data/WordPress/tmp'))

def generate_data(num=10, name='captcha'):
    X_train, y_train = dataset.generate_captcha_dataset(os.path.join(name, 'train_digit_100000'), length=4, n_samples=num, font_number=3)
    X_test, y_test = dataset.generate_captcha_dataset(os.path.join(name, 'test_digit_30000'), length=4, n_samples=int(num * 0.3), font_number=3)
    print "Train:" , len(dataset.get_image_files(os.path.join(name, 'train')))
    print "Test:", len(dataset.get_image_files(os.path.join(name, 'test')))


def rename():
    input_list=dataset.get_image_files('data/WordPress/Confirmed__')
    tmp_list = []
    while len(tmp_list) <= 100:
        choice = random.choice(input_list)
        if not choice in tmp_list:
            tmp_list.append(choice)
    for file_image in tmp_list:
        image = ImgIO.read_image(file_image)
        ImgIO.write_image(image, os.path.join('data/WordPress/test', os.path.basename(file_image)))
                          # dataset.get_save_image_name('data/WordPress/tmp',
                          #                                    os.path.basename(file_image).split('.')[0], split='_'))
        # print dataset.get_save_image_name('data/WordPress/tmp', os.path.basename(file_image).split('.')[0], split='_')
# rename()

# generate_data(num=100000)
# print len(dataset.get_image_files(os.path.join('captcha', 'train_digit_100000')))
# data, labels = dataset.load_captcha_dataset('data/zhihu', save_pkl=False)
# # X_train, y_train = dataset.load_captcha_dataset(os.path.join('captcha', 'train'), save_pkl=False)
# # X_test, y_test = dataset.load_captcha_dataset(os.path.join('captcha', 'test'), save_pkl=False)
# np.savez("data.npz",  zhihu_images=data, zhihu_labels=labels)
# print len(labels)
# images = map(ImgIO.read_image_uc, dataset.get_image_files('result/processing/jiayuan'))
# r = estimate_function_time(process2.affine_image, input_list=images[:10])
# ImgIO.show_images_list(r)


from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer

def save_processing_images(images, labels, folder, processor):
    process_images = map(processor, images)
    pklname = os.path.join('tmp', folder+'_process.pkl')
    joblib.dump((process_images, labels), pklname)
    return pklname

def save_segment_result(pklname, folder, separator):
    process_images, captcha_labels = joblib.load(pklname)
    characters, labels = [], []
    for image, label in zip(process_images, captcha_labels):
        char_images = separator(image).get_characters()
        characters.extend(char_images)
        labels.extend(label)
    new_pklname = os.path.join('tmp', folder+'_segment.pkl')
    joblib.dump((characters, labels), new_pklname)
    return new_pklname

def grid_fit_save_model(folder):
    training_folder = os.path.join('data', 'dataset', folder, 'training_set')
    images, labels = dataset.load_captcha_dataset(training_folder, save_pkl=False)
    print "Number of training set: %d" % len(labels)

    processor = None

    separator = None

    extractor = None

    t1 = time.time()
    sp_name = save_processing_images(images, labels, folder, processor)
    print "Processing time: %.4f min" % ((time.time() - t1) / 60.)

    t2 = time.time()
    ss_name = save_segment_result(sp_name, folder, separator)
    print "Segment time: %.4f min" % ((time.time() - t2) / 60.)

    t3 = time.time()
    characters, labels = joblib.load(ss_name)
    character_features = map(extractor, characters)
    print "Extract feature time: %.4f min" % ((time.time() - t3) / 60.)

    t4 = time.time()



def test_train_model(folder):
    print folder, '~~~~~~'
    train = os.path.join(folder, 'training_set')
    test = os.path.join(folder, 'testing_set')
    # image= ImgIO.read_image(os.path.join(test, '08hjv-0.png'))

    training_images, training_labels = dataset.load_captcha_dataset(train, save_pkl=False)
    testing_images, testing_labels = dataset.load_captcha_dataset(test, save_pkl=False)
    processor = ComposeProcessor(processors=[# (process2.inverse),
        # (lambda x:process.filter_scale(x, width=x.shape[1]*2, height=x.shape[0]*2),None),
        (process2.rgb_to_gray, None),
        (process2.threshold_filter, {'threshold':120}),
        (process.filer_reduce_mesh, None),
        (process2.inverse, None),
        (process2.denosie_color_filling, {'remove_size':4}),
        (process.filter_fill_holes, None),
        # (process.filter_fix_broken_characters, None)
        # (process2.extract_skeleton, None)
        # (process.filter_remove_dots, None)
    ])
    separator = ColorFillingSeparator
    # ImgIO.show_image(processor(image))
    # ImgIO.show_images_list(separator(processor(image), length=5).get_characters())
    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = KNNEngine(k=1)
    decoder = CaptchaDecoder(processor=processor, separator=separator, extractor=extractor, engine=engine, length=5)
    decoder.fit(training_images, training_labels)
    print decoder.score(testing_images, testing_labels, verbose=True)
    # decoder.save_engine(os.path.basename(folder)+'.pkl')
    save_model(decoder, os.path.join('model', os.path.basename(folder)+'.pkl'))

    p = Profiler(os.path.join('model', os.path.basename(folder)+'.pkl'))
    print p.predict(testing_images[:2])
# test_train_model(os.path.join('data', 'dataset', 'hongxiu'))


def train_hkbu_model(folder):
    print folder, '~~~~~~'
    train = os.path.join(folder, 'training_set')
    test = os.path.join(folder, 'testing_set')
    # image= ImgIO.read_image(os.path.join(test, '08hjv-0.png'))

    training_images, training_labels = dataset.load_captcha_dataset(train, save_pkl=False)
    testing_images, testing_labels = dataset.load_captcha_dataset(test, save_pkl=False)
    processor = ComposeProcessor(processors=[
        (process2.rgb_to_gray, None),
        (process2.deconvolution, None),
        (process.filter_median,None),
        (process2.otsu_filter, None),
        (process2.inverse, None),
        # (process2.reconstruction, None),
        (process2.reconstruction, None)
        # (process2.reconstruction, None),
        # (process.filter_fill_holes, None)
    ])
    separator = ColorFillingSeparator
    # ImgIO.show_image(processor(image))
    # ImgIO.show_images_list(separator(processor(image), length=5).get_characters())
    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = SVMEngine()
    decoder = CaptchaDecoder(processor=processor, separator=separator, extractor=extractor, engine=engine, length=4)
    decoder.fit(training_images, training_labels)
    print decoder.score(testing_images, testing_labels, verbose=True)
    # decoder.save_engine(os.path.basename(folder)+'.pkl')
    save_model(decoder, os.path.join('model', os.path.basename(folder)+'.pkl'))
    p = Profiler(os.path.join('model', os.path.basename(folder)+'.pkl'))
    print p.predict(testing_images[:2])
# train_hkbu_model(os.path.join('data', 'dataset', 'hkbu'))

def train_xxsy_model(folder):
    print folder, '~~~~~~'
    train = os.path.join(folder, 'training_set')
    test = os.path.join(folder, 'testing_set')
    training_images, training_labels = dataset.load_captcha_dataset(train, save_pkl=False, split_symbol='_')
    testing_images, testing_labels = dataset.load_captcha_dataset(test, save_pkl=False, split_symbol='_')
    processor = ComposeProcessor(processors=[
        #(lambda x: ColorClusterSeparator(x).kmeans_segment(), None),
        (process2.rgb_to_gray, None),
        (process2.yen_filter, None),
        (process2.median, None),
        (process.filter_fill_border_background, None),
        (process2.inverse, None),
        # (process2.closing, None),
        # (process2., None)
        # (process.filter_median,None),
        # (process2.inverse, None),
        # (process2.reconstruction, None),
        # (process2.reconstruction, None)
        # (process2.reconstruction, None),
        # (process.filter_fill_holes, None)
    ])
    separator = Separator

    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = KNNEngine(k=1)
    decoder = CaptchaDecoder(processor=processor, separator=separator, extractor=extractor, engine=engine)
    decoder.fit(training_images, training_labels)
    print decoder.score(testing_images, testing_labels, verbose=True)
    decoder.save_engine(os.path.basename(folder)+'.pkl')
# train_xxsy_model(os.path.join('data', 'dataset', 'xxsy'))



def train_color_model(folder):
    print folder, '~~~~~~'
    train = os.path.join(folder, 'tmp')#'training_set')
    # test = os.path.join(folder, 'testing_set')
    # image= ImgIO.read_image(os.path.join(test, '08hjv-0.png'))

    training_images, training_labels = dataset.load_captcha_dataset(train, save_pkl=False)
    # testing_images, testing_labels = dataset.load_captcha_dataset(test, save_pkl=False)

    processor = process2.nochange
    separator = ColorClusterSeparator

    # ImgIO.show_image(processor(image))
    # ImgIO.show_images_list(separator(processor(image), length=5).get_characters())
    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = SVMEngine()
    decoder = CaptchaDecoder(processor=processor, separator=separator, extractor=extractor, engine=engine)
    decoder.fit(training_images, training_labels)
    # print decoder.score(testing_images, testing_labels, verbose=True)
    decoder.save_engine(os.path.basename(folder)+'.pkl')
    save_model(decoder, os.path.join('model', os.path.basename(folder)+'.pkl'))

    p = Profiler(os.path.join('model', os.path.basename(folder)+'.pkl'))
    # print p.predict(testing_images[:2])
# train_color_model(os.path.join('data', 'dataset', 'wordpress'))

def train_gdgs_model(folder):
    print folder, '~~~~~~'
    train = os.path.join(folder, 'training_set')
    test = os.path.join(folder, 'testing_set')

    training_images, training_labels = dataset.load_captcha_dataset(train, save_pkl=False)
    testing_images, testing_labels = dataset.load_captcha_dataset(test, save_pkl=False)
    processor = ComposeProcessor(processors=[# (process2.inverse),
        (process2.rgb_to_gray, None),
        # (process2.otsu_filter, None),
        (process2.threshold_filter, {'threshold':125}),
        (process2.inverse, None),
    ])
    separator = Separator

    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = SVMEngine()
    decoder = CaptchaDecoder(processor=processor, separator=separator, extractor=extractor, engine=engine)
    decoder.fit(training_images, training_labels)
    print decoder.score(testing_images, testing_labels, verbose=True)
    save_model(decoder, os.path.join('model', os.path.basename(folder)+'.pkl'))

    p = Profiler(os.path.join('model', os.path.basename(folder)+'.pkl'))
    print p.predict(testing_images[:2])
# train_gdgs_model(os.path.join('data', 'dataset', 'gdgs'))

def test_color_cluster():
    testing_folder = os.path.join('data', 'WordPress', 'test')
    testing_images, testing_labels = dataset.load_captcha_dataset(testing_folder, save_pkl=False)
    p = Profiler(os.path.join('model', 'wordpress.pkl'))
    p.print_score(testing_images, testing_labels)
test_color_cluster()
