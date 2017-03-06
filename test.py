from sc_knn_decoder import *
from multiprocessing.dummy import Pool as ThreadPool
from feature import shapecontext_1
import sys

"""
def abc(a, b, c):
    return a*10000 + b*100 + c

t1 = range(1, 100, 2)
t2 = [2 for i in range(50)]
t3 = range(100, 0, -2)


s = time.clock()
tmp = map(abc, t1, t2, t3)
t = time.clock()
print "t1: %.8f" %(t - s)

s = time.clock()
tmp2 = [abc(a, b, c) for a, b, c in zip(t1, t2, t3)]
t = time.clock()
print "t2: %.8f" %(t - s)
"""
#"""
time_1 = time.time()
folder_dir = 'annotated_captchas//train2'
training_set, training_labels = dataset.load_captcha_dataset(folder_dir)
model = SC_KNN_Decoder(dataset='test_easy_digits', character_shape=(70, 70))
time_2 = time.time()
print "Load training data: %.4f s" % (time_2 - time_1)

#model.fit(training_set, training_labels)
#"""
time_3 = time.time()
test_folder = 'annotated_captchas//test2'
testing_set, testing_labels = dataset.load_captcha_dataset(test_folder)
print "Load testing data: %.4f s" % (time.time() - time_3)
#"""

"""
# print model.predict(testing_set)
# print testing_labels
# model.score(testing_set, testing_labels)

def save_result(index):
    lower = index
    upper = min(index+5, len(testing_labels))
    number = index / 5
    model.fast_score(testing_set[lower:upper], testing_labels[lower:upper], mode='save', paras='multi_fast'+str(number))

pool = ThreadPool(10)
pool.map(save_result, range(300, 400))
pool.close()
pool.join()
#"""

