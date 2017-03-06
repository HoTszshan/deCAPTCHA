from __future__ import print_function
from sc_knn_decoder import *
import csv
import re

#"""
folder = '_result_time'
file_path = os.path.join(folder, 'test_output_3.txt')
pattern_fast_time = re.compile(r'It takes (.*?) min to fast predict a image:\t')# \d{4}\n')
pattern_predict_t = re.compile(r'It takes (.*?) min to predict a image. ')
pattern_label = re.compile(r'Label\: (\d*)\tFast_predict\: (\d*)\tPredict\: (\d*)')

with open(file_path, "r") as txt_file:
    result = txt_file.read()
    fast_time = re.findall(pattern_fast_time, result)
    pred_time = re.findall(pattern_predict_t, result)
    label_items = re.findall(pattern_label, result)
    result_list = []
    print(str(len(fast_time))+' '+str(len(pred_time)) + ' ' + str(len(label_items)))

    #"""
    for label, t1, t2 in zip(label_items, fast_time, pred_time):
        result_list.append([label[0], label[1], label[2], t1, t2])

    with open(os.path.join(folder, 'test_minutes_3.csv'), 'wb') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(result_list)
    #"""


#"""

start_time = time.time()
"""
start_test_time = time.time()
folder_dir = 'annotated_captchas//test2'
testing_set, testing_labels = dataset.load_captcha_dataset(folder_dir)

print("It takes %.4f s to load data." % (time.time() - start_time))
model = Digit_CNN_Decoder('cnn_ez', character_shape=(28,28), length=4)

#for index in range(0, len(testing_labels), 10):
#    upper = min(index+10, len(testing_labels))
#    print(model.evaluate(testing_set[index:upper], testing_labels[index:upper]))
print(model.evaluate(testing_set, testing_labels))

finish_time = time.time()
print("Test time: %.4f" % (finish_time - start_time))
"""
"""
folder = "_result_time"
file_list = [os.path.join(folder,f) for f in os.listdir(folder) if f.endswith('.csv')]
result_list_1 = ['fast_test' + file_name.split('fast_test')[1] for file_name in file_list if len(file_name.split('fast_test')) > 1]
result_list_2 = ['fast_prune' + file_name.split('fast_prune')[1] for file_name in file_list if len(file_name.split('fast_prune')) > 1]
result_list = []
result_list.extend(result_list_1)
result_list.extend(result_list_2)
result = []#[['label', 'predict', 'fast_predict', 'pre_matching', 'fast_matching', 'predict(success rate)', 'fast(success rate)']]
#result = [['label', 'predict', 'fast_predict', 'pre_matching', 'fast_matching', 'predict(success rate)', 'fast(success rate)']]
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
#"""
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
"""
#result = sorted(result, lambda x, y: cmp(x[0], y[0]))

#voting_sorted = [v for v in sorted(voting_dict.items(), lambda x, y: cmp(x[1], y[1]))]

with open(folder + '/' + 'test_result2' + '.csv', 'wb') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(result)

finish_time = time.time()
print "Time: ", finish_time - start_time, 's'
"""
"""
for i in range(len(digit_dict)):
    print digit_dict.keys()[i],'\t', digit_dict[i], '\t\t', wrong_dict.keys()[i], '\t',
    print len(wrong_dict[i]) if type(wrong_dict[i]) == list else wrong_dict[i], '\t',
    print wrong_dict[i]
#"""



"""
folder = "_result_time"
with open(folder + '/' + 'test_result_2' + '.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    result = []
    for row in reader:
        result.append([float(number) for number in row])

    output = zip(*result)
    output_list = []
    for i in range(3, len(output)):
        output_list.append([sum(output[i]) / float(len(output[0]))])
"""
