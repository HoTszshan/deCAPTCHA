from lib import imgio as ImgIO
from lib import dataset

from PIL import Image
from scipy import sparse
from skimage import feature as ski_feature

from sc_knn_decoder import *
import csv

start_time = time.time()

"""
file_path = "test_result_1.txt"

with open(file_path, "r") as txt_file:
    result_list = txt_file.readlines()

test_result = []
for result in result_list:
    result = result.split('\n')[0]
    split_result = result.split(' ')
    predict_time, label = split_result[2], split_result[-1]
    test_result.append([predict_time, label])

test_folder = 'annotated_captchas//test'
testing_set, testing_labels = dataset.load_captcha_dataset(test_folder)

result_length = len(test_result)
result_list = []
print result_length
for result, label in zip(test_result, testing_labels[:result_length]):
    predict_time, predict_label = result
    global_matching = 1 if predict_label == label else 0
    local_matching = sum([1 for i in range(min(len(predict_label), len(label))) if predict_label[i] == label[i]])
    result_list.append([predict_time, predict_label, label, global_matching, local_matching])
            #[str(predict_time), str(predict_label), str(label),
            #             str(global_matching), str(local_matching)])

with open('test_result.csv', 'wb') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(result_list)
"""

folder = "_result_time"
file_list = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.csv')]
result_list_1 = ['fast_test' + file_name.split('fast_test')[1] for file_name in file_list if len(file_name.split('fast_test')) > 1]
result_list_2 = ['fast_prune' + file_name.split('fast_prune')[1] for file_name in file_list if len(file_name.split('fast_prune')) > 1]
result_list = []
result_list.extend(result_list_1)
result_list.extend(result_list_2)


result = []#[['label', 'predict', 'fast_predict', 'pre_matching', 'fast_matching', 'predict(success rate)', 'fast(success rate)']]
digit_dict = {}.fromkeys(np.linspace(0,9, 10, dtype=np.uint8), 0)
wrong_dict = {}.fromkeys(np.linspace(0,9, 10, dtype=np.uint8), 0)

for result_file in result_list:
    file_path = folder + '/' +  result_file
    with open(file_path, 'rb') as csvfile:
        result_reader = csv.reader(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
        for row in result_reader:
            label, pre_label, fast_label = row[0], row[1], row[2]
            char_success = [pre_label[i] for i in range(len(label)) if label[i] == pre_label[i]]
            fast_success = [fast_label[i] for i in range(len(label)) if label[i] == fast_label[i]]
            c_matching = 1 if len(char_success) == len(label) else 0 #0 if matching == 'True' else 1
            f_matching = 1 if len(fast_success) == len(label) else 0 #0 if float(fast_rate) < 1.0 else 1
            result.append([label, pre_label, fast_label, c_matching, f_matching, str(len(char_success) / float(len(label))),
                           str(len(fast_success) / float(len(label)))])#fast_rate])
            """
            for i in label: digit_dict[int(i)] += 1
            for i in range(len(label)):
                if not label[i] == fast_label[i]:
                    if wrong_dict[int(label[i])] == 0:
                        wrong_dict[int(label[i])] = []
                        wrong_dict[int(label[i])].append((label[i], fast_label[i]))
                    else:
                        wrong_dict[int(label[i])].append((label[i], fast_label[i]))
            #"""

result = sorted(result, lambda x, y: cmp(x[0], y[0]))


#voting_sorted = [v for v in sorted(voting_dict.items(), lambda x, y: cmp(x[1], y[1]))]

with open(folder + '/' + 'test_result2' + '.csv', 'wb') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(result)

finish_time = time.time()
print "Time: ", finish_time - start_time, 's'
"""
for i in range(len(digit_dict)):
    print digit_dict.keys()[i],'\t', digit_dict[i], '\t\t', wrong_dict.keys()[i], '\t',
    print len(wrong_dict[i]) if type(wrong_dict[i]) == list else wrong_dict[i], '\t',
    print wrong_dict[i]
#"""