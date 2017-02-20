from sc_knn_decoder import *
from feature import shapecontext_1
import sys

sys.setrecursionlimit(10000)

model = SC_KNN_Decoder(dataset='test_easy_digits', character_shape=(70, 70), sys='XOS')
folder_dir = 'annotated_captchas//test2'
testing_set, testing_labels = dataset.load_captcha_dataset(folder_dir)
"""
for index in range(len(testing_labels)-80,0, -5):
    #upper = min(index-10, len(testing_labels))
    below = max(index-5, 0)
    number = (len(testing_labels) - index) / 5
    model.fast_score(testing_set[index:below:-1], testing_labels[index:below:-1], mode='save', paras='fast_prune'+str(number))
    #model.fast_score(testing_set[index:upper], testing_labels[index:upper], mode='save', paras='fast_prune'+str(index))
"""


#"""

def save_result(index):
    below = max(index-5, 0)
    number = (len(testing_labels) - index) / 5
    model.fast_score(testing_set[index:below:-1], testing_labels[index:below:-1], mode='save', paras='multi_fast'+str(number))

pool = ThreadPool(10)
pool.map(save_result, range(len(testing_labels)-80,0, -5))
pool.close()
pool.join()

#"""



"""
test_img_folder = 'annotated_captchas//'
image_list = ['0120-0.jpg']
for file in image_list:
    test_img_file = test_img_folder + file
    test_img = imgio.read_img_uc(test_img_file)

    test_pre_img = pre_processing_digit(test_img)
    separater = CharacterSeparator(test_pre_img, character_shape=(70,70))
    separater.segment_process()
    #separater.show_split_objects()
#"""
"""
model = SC_KNN_Decoder(dataset='digits', character_shape=(70, 70), sys='XOS')
model.predict([test_img])
"""
"""
test_folder = 'annotated_captchas//test1'
testing_set, testing_labels = dataset.load_captcha_dataset(test_folder)
for label in testing_labels:
    print label

def get_image(path):
    img = imgio.read_img_uc(path)
    img = process.filter_scale(img, width=70, height=70)
    img = process.filter_threshold(img, threshold=128)
    return img
"""
"""
#for i in range(10):
img_file_1 = 'digit\\0\\10.jpg'
img_file_2 = 'digit\\0\\15.jpg' # + str(i) + '\\15.jpg'
img_1 = get_image(img_file_1)
img_2 = get_image(img_file_2)

tmp_time_1 = time.time()
sc_1 = ShapeContext(img_1)
sc_2 = ShapeContext(img_2)
tmp_time_2 = time.time()
matcher = ShapeContextMatcher(sc_1, sc_2)
matcher.calculate_shape_distance(l_0=1000)

tmp_time_3 = time.time()


tmp_time_n1 = time.time()
t_sc_1 = shapecontext_1.ShapeContext(img_1)

t_sc_2 = shapecontext_1.ShapeContext(img_2)
tmp_time_n2 = time.time()
t_matcher = shapecontext_1.ShapeContextMatcher(t_sc_1, t_sc_2)
t_matcher.calculate_shape_distance()


tmp_time_n3 = time.time()

tmp_time = time.time()
gsc_1 = GeneralizedShapeContext(img_1, sample='rsc', sample_params=0.3)
print "Generalized Shape Context Time:", time.time() - tmp_time
#gsc_2 = GeneralizedShapeContext(img_2)

print "Construct original sc:", tmp_time_n2 - tmp_time_n1, '\t',
print "Original matching time:", tmp_time_n3 - tmp_time_n2, '\t',
print "Total time:", tmp_time_n3 - tmp_time_n1

print "Construct updated sc:", tmp_time_2 - tmp_time_1, '\t',
print "Updated matching time:", tmp_time_3 - tmp_time_2, '\t',
print "Total time:", tmp_time_3 - tmp_time_1
#"""

"""
def f(x):
    time.sleep(1)
    return x ** 2

def test_1(): #mylist):
    mylist = [1, 4, 9]
    result = [f(i) for i in mylist]
    return result

def test_2(): #mylist):
    mylist = [1, 4, 9]
    pool = ThreadPool(5)
    result = pool.map(f, mylist)
    pool.close()
    pool.join()
    return result

if __name__ == '__main__':
    t1 = Timer("test_1()", "from __main__ import test_1")
    t2 = Timer("test_2()", "from __main__ import test_2")
    print t1.timeit(10)
    print t2.timeit(10)
#"""