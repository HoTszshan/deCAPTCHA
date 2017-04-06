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


    extractor = ComposeExtractor([ScaleExtract(position_brightness)])
    engine = SVMEngine()
    decoder = CaptchaDecoder(processor=processor, separator=ColorFillingSeparator, extractor=extractor, engine=engine)
    decoder.fit(training_images, training_labels, grid_search = True)
    decoder.score(testing_images, testing_labels, verbose=True)

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

# img2 = filters.wiener(img1)
# print type(img2)
# ImgIO.show_images_list([img1, img2])
# f = 'data/hkbu_1000/2d75-0.jpg'#'data/ez-gimpy/adjust.jpg'
f = 'data/misc/' #'data/pconline_1000/'

image = ImgIO.read_image('data/misc/92RY-0.tiff')
# image = ImgIO.read_image('data/hkgolden/8.jpg')
# print np.unique(process2.rgb_to_gray(image))
# ImgIO.show_images_list([image, process2.rgb_to_gray(image), process2.otsu_filter(process2.rgb_to_gray(image)),
#                         (process2.smooth(image)),
#                         process2.median(process2.otsu_filter(process2.rgb_to_gray(image)))])
# tmp = process2.median(image)
# ImgIO.show_images_list([image, tmp, process2.otsu_filter(tmp)])
def test_c_segment(filename):
    image = ImgIO.read_image(filename)#'data/pconline_1000/5apnp-0.jpg')
    separator = ColorClusterSeparator(image, n_cluster=16)
    separator.segment_process()
    # separator.display_kmeans_result()
    # separator.display_segment_result()
    ImgIO.show_images_list(separator.get_characters())
# test_c_segment('data/word-press/15.jpg')
estimate_function_time(test_c_segment, input_list=dataset.get_image_files('data/word-press'))
def generate_data(num=10, name='captcha'):
    X_train, y_train = dataset.generate_captcha_dataset(os.path.join(name, 'train_digit_100000'), length=4, n_samples=num, font_number=3)
    X_test, y_test = dataset.generate_captcha_dataset(os.path.join(name, 'test_digit_30000'), length=4, n_samples=int(num * 0.3), font_number=3)
    print "Train:" , len(dataset.get_image_files(os.path.join(name, 'train')))
    print "Test:", len(dataset.get_image_files(os.path.join(name, 'test')))

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





'''
This example uses a convolutional stack followed by a recurrent stack
and a CTC logloss function to perform optical character recognition
of generated text images. I have no evidence of whether it actually
learns general shapes of text, or just is able to recognize all
the different fonts thrown at it...the purpose is more to demonstrate CTC
inside of Keras.  Note that the font list may need to be updated
for the particular OS in use.

This starts off with 4 letter words.  For the first 12 epochs, the
difficulty is gradually increased using the TextImageGenerator class
which is both a generator class for test/train data and a Keras
callback class. After 20 epochs, longer sequences are thrown at it
by recompiling the model to handle a wider image and rebuilding
the word list to include two words separated by a space.

The table below shows normalized edit distance values. Theano uses
a slightly different CTC implementation, hence the different results.

            Norm. ED
Epoch |   TF   |   TH
------------------------
    10   0.027   0.064
    15   0.038   0.035
    20   0.043   0.045
    25   0.014   0.019

This requires cairo and editdistance packages:
pip install cairocffi
pip install editdistance

Created by Mike Henry
https://github.com/mbhenry/
'''
"""
import os
import itertools
import re
import datetime
# import cairocffi as cairo
import editdistance
import numpy as np
from scipy import ndimage
import pylab
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.preprocessing import image
import keras.callbacks


OUTPUT_DIR = 'image_ocr'

np.random.seed(55)


# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

def speckle(img):
    severity = np.random.uniform(0, 0.6)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text(text, w, h, rotate=False, ud=False, multi_fonts=False):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    with cairo.Context(surface) as context:
        context.set_source_rgb(1, 1, 1)  # White
        context.paint()
        # this font list works in Centos 7
        if multi_fonts:
            fonts = ['Century Schoolbook', 'Courier', 'STIX', 'URW Chancery L', 'FreeMono']
            context.select_font_face(np.random.choice(fonts), cairo.FONT_SLANT_NORMAL,
                                     np.random.choice([cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]))
        else:
            context.select_font_face('Courier', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        context.set_font_size(25)
        box = context.text_extents(text)
        border_w_h = (4, 4)
        if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
            raise IOError('Could not fit string into image. Max char count is too large for given image width.')

        # teach the RNN translational invariance by
        # fitting text box randomly on canvas, with some room to rotate
        max_shift_x = w - box[2] - border_w_h[0]
        max_shift_y = h - box[3] - border_w_h[1]
        top_left_x = np.random.randint(0, int(max_shift_x))
        if ud:
            top_left_y = np.random.randint(0, int(max_shift_y))
        else:
            top_left_y = h // 2
        context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
        context.set_source_rgb(0, 0, 0)
        context.show_text(text)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (h, w, 4)
    a = a[:, :, 0]  # grab single channel
    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)
    if rotate:
        a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
    a = speckle(a)

    return a


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = range(stop_ind)
    np.random.shuffle(a)
    a += range(stop_ind, len_val)
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('shuffle_mats_or_lists only supports '
                            'numpy.array and list objects')
    return ret


def text_to_labels(text, num_classes):
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret


# only a-z and space..probably not to difficult
# to expand to uppercase and symbols

def is_valid_str(in_str):
    search = re.compile(r'[^a-z\ ]').search
    return not bool(search(in_str))


# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, monogram_file, bigram_file, minibatch_size,
                 img_w, img_h, downsample_factor, val_split,
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.monogram_file = monogram_file
        self.bigram_file = bigram_file
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return 28

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.5):
        assert max_string_len <= self.absolute_max_string_len
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = [''] * self.num_words
        tmp_string_list = []
        self.max_string_len = max_string_len
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words

        # monogram file is sorted by frequency in english speech
        with open(self.monogram_file, 'rt') as f:
            for line in f:
                if len(tmp_string_list) == int(self.num_words * mono_fraction):
                    break
                word = line.rstrip()
                if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
                    tmp_string_list.append(word)

        # bigram file contains common word pairings in english speech
        with open(self.bigram_file, 'rt') as f:
            lines = f.readlines()
            for line in lines:
                if len(tmp_string_list) == self.num_words:
                    break
                columns = line.lower().split()
                word = columns[0] + ' ' + columns[1]
                if is_valid_str(word) and \
                        (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(word)
        if len(tmp_string_list) != self.num_words:
            raise IOError('Could not pull enough words from supplied monogram and bigram files. ')
        # interlace to mix up the easy and hard words
        self.string_list[::2] = tmp_string_list[:self.num_words // 2]
        self.string_list[1::2] = tmp_string_list[self.num_words // 2:]

        for i, word in enumerate(self.string_list):
            self.Y_len[i] = len(word)
            self.Y_data[i, 0:len(word)] = text_to_labels(word, self.get_output_size())
            self.X_text.append(word)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(0, size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = 1
                source_str.append('')
            else:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func(self.X_text[index + i])[0, :, :].T
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        self.build_word_list(16000, 4, 1)
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                  rotate=False, ud=False, multi_fonts=False)

    def on_epoch_begin(self, epoch, logs={}):
        # rebind the paint function to implement curriculum learning
        if epoch >= 3 and epoch < 6:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=False, ud=True, multi_fonts=False)
        elif epoch >= 6 and epoch < 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=False, ud=True, multi_fonts=True)
        elif epoch >= 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=True, ud=True, multi_fonts=True)
        if epoch >= 21 and self.max_string_len < 12:
            self.build_word_list(32000, 12, 0.5)


# the actual loss calc occurs here despite it not being
# an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = ''
        for c in out_best:
            if c >= 0 and c < 26:
                outstr += chr(c + ord('a'))
            elif c == 26:
                outstr += ' '
        ret.append(outstr)
    return ret


class VizCallback(keras.callbacks.Callback):

    def __init__(self, run_name, test_func, text_img_gen, num_display_words=6):
        self.test_func = test_func
        self.output_dir = os.path.join(
            OUTPUT_DIR, run_name)
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def show_edit_distance(self, num):
        num_left = num
        mean_norm_ed = 0.0
        mean_ed = 0.0
        while num_left > 0:
            word_batch = next(self.text_img_gen)[0]
            num_proc = min(word_batch['the_input'].shape[0], num_left)
            decoded_res = decode_batch(self.test_func, word_batch['the_input'][0:num_proc])
            for j in range(0, num_proc):
                edit_dist = editdistance.eval(decoded_res[j], word_batch['source_str'][j])
                mean_ed += float(edit_dist)
                mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
            num_left -= num_proc
        mean_norm_ed = mean_norm_ed / num
        mean_ed = mean_ed / num
        print('\nOut of %d samples:  Mean edit distance: %.3f Mean normalized edit distance: %0.3f'
              % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        self.show_edit_distance(256)
        word_batch = next(self.text_img_gen)[0]
        res = decode_batch(self.test_func, word_batch['the_input'][0:self.num_display_words])
        if word_batch['the_input'][0].shape[0] < 256:
            cols = 2
        else:
            cols = 1
        for i in range(self.num_display_words):
            pylab.subplot(self.num_display_words // cols, cols, i + 1)
            if K.image_data_format() == 'channels_first':
                the_input = word_batch['the_input'][i, 0, :, :]
            else:
                the_input = word_batch['the_input'][i, :, :, 0]
            pylab.imshow(the_input.T, cmap='Greys_r')
            pylab.xlabel('Truth = \'%s\'\nDecoded = \'%s\'' % (word_batch['source_str'][i], res[i]))
        fig = pylab.gcf()
        fig.set_size_inches(10, 13)
        pylab.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        pylab.close()


def train(run_name, start_epoch, stop_epoch, img_w):
    # Input Parameters
    img_h = 64
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    fdir = os.path.dirname(get_file('wordlists.tgz',
                                    origin='http://www.isosemi.com/datasets/wordlists.tgz', untar=True))

    img_gen = TextImageGenerator(monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
                                 bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
                                 minibatch_size=32,
                                 img_w=img_w,
                                 img_h=img_h,
                                 downsample_factor=(pool_size ** 2),
                                 val_split=words_per_epoch - val_words
                                 )
    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal GRUs
    # GRU seems to work as well, if not better than LSTM:
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    gru1_merged = add([gru_1, gru_1b])
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

    # transforms RNN output to character activations:
    inner = Dense(img_gen.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([gru_2, gru_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    print('sssssss')
    Model(inputs=input_data, outputs=y_pred).summary()
    print('yyyy')
    labels = Input(name='the_labels', shape=[img_gen.absolute_max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        model.load_weights(weight_file)
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function([input_data], [y_pred])

    viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

    model.fit_generator(generator=img_gen.next_train(), steps_per_epoch=(words_per_epoch - val_words),
                        epochs=stop_epoch, validation_data=img_gen.next_val(), validation_steps=val_words,
                        callbacks=[viz_cb, img_gen], initial_epoch=start_epoch)


if __name__ == '__main__':
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    train(run_name, 0, 20, 128)
    # increase to wider images and start at epoch 20. The learned weights are reloaded
    train(run_name, 20, 25, 512)
"""