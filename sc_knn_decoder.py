# Use KNN classifier
# Base on the Shape Context feature descriptor

from feature.shapecontext import *
from sklearn.neighbors import *
from lib import process
from lib import dataset
from lib import segment as Seg
from lib import imgio as ImgIO
import copy
import dill
import pickle
from klepto.archives import dir_archive
import os
import csv
from multiprocessing.dummy import Pool as ThreadPool

character_width = 70
character_height = 70
sc_sampling_num = 100

# basic operation
def np_array_to_list(img):
    #return [img[i][j] for i in range(h) for j in range(w)]
    return img.reshape(img.size)

def list_to_np_array(ilist, w, h):
    """
    img = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            img[i][j] = ilist[i*w+j]
    return img
    """
    return ilist.reshape((w, h))

# pre-processing function
def pre_processing_digit(image):
    img = process.filter_inverse(image)
    img = process.filter_reduce_lines(img, median=200)
    img = process.filter_mean_smooth(img)
    img = process.filter_threshold_RGB(img, threshold=150)
    img = process.filter_fix_broken_characters(img)
    img = process.filter_fill_holes(img)
    img = process.filter_fix_broken_characters(img)
    #img = process.filter_erosion(img, window_size=2)
    #img = process.filter_dilation(img, window_size=3)
    # ImgIO.show_img(img)
    return img

def post_processing_digit(image):
    character_img = process.filter_reduce_lines(image, median=200)
    character_img = process.filter_erosion(character_img)
    character_img = process.filter_remove_dots(character_img)
    # character_img = process.filter_mean_smooth(character_img)
    character_img = process.filter_dilation(character_img)
    return character_img

def process_functions(image, func_list):
    img = copy.deepcopy(image)
    for func in func_list:
        if len(func) == 1:
            img = func(img)
        elif len(func) == 2:
            img = func(img, func[1])
        elif len(func) == 3:
            img = func(img, func[1], func[2])
        else:
            raise ValueError('too many parameters!')
    return img


# metric function
def ShapeContextDistance(x, y, width, height, sample_number):
    img_x = list_to_np_array(x, width, height)
    img_y = list_to_np_array(y, width, height)
    sc_x = ShapeContext(img_x, sample_num=sample_number)
    sc_y = ShapeContext(img_y, sample_num=sample_number)
    return ShapeContextMatcher(sc_x, sc_y).calculate_shape_distance()

# classifier
def knn_engine(k=3, shape=(character_height, character_width), sample_num=sc_sampling_num):
    metric_func = lambda x, y: ShapeContextDistance(x, y, shape[1], shape[0], sample_num)
    return KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree',
                                metric=metric_func, n_jobs=-1)


class SC_KNN_Decoder(object):
    def __init__(self, dataset, character_shape, sample_number=sc_sampling_num, sys='Windows', *args, **kwargs):
        #process_func=pre_processing_digit): #dataset_name='Digit'):
        self.engine = knn_engine(shape=character_shape, sample_num=sample_number)
        self.pre_processor = pre_processing_digit
        self.dataset_name = dataset
        self.character_shape = character_shape
        self.sample_number=sample_number
        if sys == 'Windows':
            self.sys_split = '\\'
        else:
            self.sys_split = '/'

    def fit(self, X, y):
        # Segment
        characters, labels = self.segment_captchas_dataset(X, y, mode='save')
        #pickle.dump((characters, labels), open(self.dataset_name + self.sys_split + "tmp_data.txt", "wb"))
        #characters, labels = pickle.load(open("tmp_data.txt", "rb"))

        # Get known shape
        known_gsc, known_labels = self.figure_known_shapes(characters, labels)
        # fast prune
        start_fast_prune = time.clock()
        self.fast_pruner = FastPruner(known_labels, known_gsc)
        key_list = self.calculate_fast_prune_keys(self.fast_pruner, characters)
        self.fast_engine = self.get_fast_prune_engines(characters, labels, key_list)
        finish_fast_prune = time.clock()
        print "It takes %.4f min to fast pruner." % ((finish_fast_prune - start_fast_prune ) / 60.0)

        # Train
        start_training_time = time.time()
        char_training_list = [np_array_to_list(c) for c in characters]
        self.engine.fit(char_training_list, labels)
        finish_training_time = time.time()
        print "It takes %.4f min to train." % ((finish_training_time - start_training_time) / 60.0) + ('\t' +
            "The size of training set is %4d" % len(labels))

        # Save the model
        start_save_time = time.clock()
        model_file = self.dataset_name + self.sys_split + 'model' + '.pkl'
        pickle.dump(self.engine, open(model_file, "wb"))
        fast_prune_file = self.dataset_name + self.sys_split + 'fast_pruner' + '.pkl'
        pickle.dump(self.fast_pruner, open(fast_prune_file, "wb"))
        prune_engine_file = self.dataset_name + self.sys_split + 'fast_model'
        prune_engine = dir_archive(prune_engine_file, self.fast_engine, serialized=True)
        prune_engine.dump()
        finish_save_time = time.clock()
        print "It takes %.4f s to save a model." % (finish_save_time - start_save_time)
        #"""

    def __make_prediction(self, image):
        def predict_digit(c_list):
            return self.engine.predict([c_list])[0]
        """
        start_time = time.clock()
        img = self.pre_processor(image)
        separator = Seg.CharacterSeparator(img, self.character_shape)
        img_list = separator.segment_process()
        char_testing_list = [np_array_to_list(c) for c in img_list]
        pool = ThreadPool(4)
        result = pool.map(predict_digit, char_testing_list)
        pool.close()
        pool.join()
        finish_time = time.clock()
        print "It takes %.4f min to predict a image: \t" % ((finish_time - start_time) / 60.0) + ''.join(result)
        """
        start_time = time.clock()
        img = self.pre_processor(image)
        separator = Seg.CharacterSeparator(img, self.character_shape)
        img_list = separator.segment_process()
        char_testing_list = [np_array_to_list(c) for c in img_list]
        result = self.engine.predict(char_testing_list)
        finish_time = time.clock()
        print "It takes %.4f min to predict a image." % ((finish_time - start_time) / 60.0), ''.join(result)
        #"""
        return result #''.join(result)

    def __make_fast_prediction(self, image):
        start_time = time.clock()
        img = self.pre_processor(image)
        separator = Seg.CharacterSeparator(img, self.character_shape)
        img_list = separator.segment_process()
        #result = []
        #"""
        def fast_predict_digit(img):
            tag = self.__get_fast_prune_tag(self.fast_pruner, img)
            print "Prune Tag:\t" + ''.join(tag)
            if len(tag) == 1:
                return ''.join(tag)
            else:
                # predict
                tag_str = ''.join(tag)
                if tag_str in self.fast_engine.keys():
                    return self.fast_engine[tag_str].predict([np_array_to_list(img)])[0]
                else:
                    # self.derive_fast_key_engine(tag_str)
                    if len(tag_str) <= 3:
                        self.fast_engine[tag_str] = self.derive_fast_key_engine(tag_str)
                        prune_engine_file = self.dataset_name + self.sys_split + 'fast_model'
                        prune_engine = dir_archive(prune_engine_file, self.fast_engine, serialized=True)
                        prune_engine.dump()
                        print "Save new fast engine!"
                        return self.fast_engine[tag_str].predict([np_array_to_list(img)])[0]
                    else:
                        return self.engine.predict([np_array_to_list(img)])[0]
        """
        for img in img_list:
            # fast prune
            tag = self.__get_fast_prune_tag(self.fast_pruner, img)
            print "Prune Tag:\t" + ''.join(tag)
            if len(tag) == 1:
                result.extend(''.join(tag))
            else:
                # predict
                tag_str = ''.join(tag)
                if tag_str in self.fast_engine.keys():
                    result.extend(self.fast_engine[tag_str].predict([np_array_to_list(img)]))
                else:
                    # self.derive_fast_key_engine(tag_str)
                    if len(tag_str) <= 3:
                        self.fast_engine[tag_str] = self.derive_fast_key_engine(tag_str)
                        result.extend(self.fast_engine[tag_str].predict([np_array_to_list(img)]))
                        prune_engine_file = self.dataset_name + self.sys_split + 'fast_model'
                        prune_engine = dir_archive(prune_engine_file, self.fast_engine, serialized=True)
                        prune_engine.dump()
                        print "Save new fast engine!"
                    else:
                        result.extend(self.engine.predict([np_array_to_list(img)]))
        #"""
        #"""
        pool = ThreadPool(4)
        result = pool.map(fast_predict_digit, img_list)
        pool.close()
        pool.join()
        #"""
        finish_time = time.clock()
        print "It takes %.4f min to fast predict a image:\t" % ((finish_time - start_time) / 60.0)  + ''.join(result)
        return ''.join(result)

    def predict(self, x):
        model_file = self.dataset_name + self.sys_split + 'model' + '.pkl'
        if os.path.isfile(model_file):
            # print model_file
            self.engine = pickle.load(open(model_file, "rb"))
        if not hasattr(x, '__iter__'):
            return self.__make_prediction(x)
        else:
            result = [self.__make_prediction(image) for image in x]
            return result

    def fast_predict(self, x):
        model_file = self.dataset_name + self.sys_split + 'model' + '.pkl'
        fast_prune_file = self.dataset_name + self.sys_split + 'fast_pruner' + '.pkl'
        prune_engine_file = self.dataset_name + self.sys_split + 'fast_model'
        if os.path.isfile(fast_prune_file):
            self.fast_pruner = pickle.load(open(fast_prune_file, "rb"))
            #self.fast_pruner, self.fast_engine = pickle.load(open(fast_model_file, "rb"))
            if os.path.exists(prune_engine_file):
                self.fast_engine = dir_archive(prune_engine_file, {}, serialized=True)
                self.fast_engine.load()
        if os.path.isfile(model_file):
            self.engine = pickle.load(open(model_file, "rb"))
        if not hasattr(x, '__iter__'):
            return self.__make_fast_prediction(x)
        else:
            result = [self.__make_fast_prediction(image) for image in x]
            return result

    def score(self, data, labels):
        pred_labels = self.predict(data)
        match_captchas_error = []
        match_characters_error = []
        for i in range(len(labels)):
            if not pred_labels[i] == labels[i]:
                match_captchas_error.append((pred_labels[i], labels[i]))
                wrong_char = [(pred_labels[i][j], labels[i][j]) for j in range(len(labels[i]))
                              if not pred_labels[i][j] == labels[i][j]]
                match_characters_error.extend(wrong_char)
        captchas_error = float(len(match_captchas_error)) / float(len(labels))
        characters_error = float(len(match_characters_error)) / float(len(labels) * len(labels[0]))
        for error in match_captchas_error:
            print error, '\t',
        print ''
        for c_error in match_characters_error:
            print c_error, '\t',
        print ''
        print captchas_error, characters_error
        return captchas_error, characters_error

    def fast_score(self, data, labels, mode='show', paras=None ):
        out_file = '_result_time' + self.sys_split + paras + '.csv' if paras else \
            '_result_time' + self.sys_split + 'fast_score.csv'
        result_list = [] if mode == 'save' else None

        for image, label in zip(data, labels):
            fast_pre_label = self.fast_predict([image])[0]
            pre_label = self.predict([image])[0]
            sta_char_error = [(i,label[i], pre_label[i]) for i in range(len(label)) if not label[i] == pre_label[i]]
            fast_char_error = [(i, label[i], fast_pre_label[i]) for i in range(len(label)) if not label[i] == fast_pre_label[i]]
            print_str = 'Test a CAPTCHA: %%% ' + 'Label: ' + label + '\t' \
                        + 'Fast_predict: ' + fast_pre_label + '\t' \
                        + 'Predict: ' + pre_label + '\t' \
                        + 'Standard Success Rate: ' + str(1 - len(sta_char_error) / float(len(label))) + '\t'  \
                        + 'FastPrune Success Rate: ' + str(1 - len(fast_char_error) / float(len(label)))
            #if mode == 'show': print print_str
            print print_str
            if type(result_list) == list:
                matching = False if len(sta_char_error) else True
                result_list.append([label, pre_label, fast_pre_label, matching,
                                    str(1 - len(sta_char_error) / float(len(label))),
                                    str(1 - len(fast_char_error) / float(len(label)))])
                """
                if len(sta_char_error):
                    separator = Seg.CharacterSeparator(self.pre_processor(image), self.character_shape)
                    img_list = separator.segment_process()
                    ImgIO.show_img_list(img_list)
                    for index, char_l, pre_l in fast_char_error:
                        ImgIO.show_img(img_list[index])
                """

        if mode == 'save':
            with open(out_file, 'wb') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerows(result_list)

    def get_params(self, *args, **kwargs):
        return self.engine.get_params(*args, **kwargs)

    def segment_captchas_dataset(self, X, y, mode='read'):
        characters = []
        labels = []
        start_seg_time = time.clock()
        for image, param_labels in zip(X, y):
            img = self.pre_processor(image)

            separator = Seg.CharacterSeparator(img, self.character_shape)
            img_list = separator.segment_process()
            # img_list = [process.filter_erosion(img) for img in img_list]
            # Save segment result
            if mode == 'save':
                separator.save_segment_result(self.dataset_name+ self.sys_split +'segment', param_labels, self.sys_split)
            characters.extend(img_list)
            labels.extend(param_labels)
        finish_seg_time = time.clock()
        print "It takes %.4f s to segment." % (finish_seg_time - start_seg_time)
        return characters, labels

    def figure_known_shapes(self, char_images, labels):
        start_label_time = time.clock()
        self.save_split_characters(char_images, labels)
        # self.get_same_label_image_list_byIO()
        img_label_list = self.get_same_label_image_list(char_images, labels)

        # Define known shape
        #"""
        aver_known_images, aver_known_labels = self.get_label_average_image(img_label_list)
        aver_known_images = [process.filter_threshold_RGB(img, threshold=150) for img in aver_known_images]
        pickle.dump((aver_known_images, aver_known_labels), open(self.dataset_name + self.sys_split + "average_image.pkl", "wb"))
        kmed_known_images, kmed_known_labels = self.get_label_K_medoids_image(img_label_list)
        pickle.dump((kmed_known_images, kmed_known_labels), open(self.dataset_name + self.sys_split + "k-medoids.pkl", "wb"))
        #"""
        #aver_known_images, aver_known_labels = pickle.load(open(self.dataset_name + self.sys_split + "average_image.pkl", "rb"))
        #kmed_known_images, kmed_known_labels = pickle.load(open(self.dataset_name + self.sys_split + "k-medoids.pkl", "rb"))
        known_images, known_labels= ([], [])
        aver_known_labels = [label + 'a' for label in aver_known_labels]
        kmed_known_labels = [label + 'k' for label in kmed_known_labels]
        known_images.extend(aver_known_images)
        known_images.extend(kmed_known_images)
        known_labels.extend(aver_known_labels)
        known_labels.extend(kmed_known_labels)
        known_images_gsc = [GeneralizedShapeContext(image) for image in known_images]
        finish_label_time = time.clock()
        print "It takes %.4f s to get known image." % (finish_label_time - start_label_time)
        return known_images_gsc, known_labels

    def save_split_characters(self, char_imgs, labels):
        for image, label in zip(char_imgs, labels):
            label_folder = self.dataset_name + self.sys_split + label
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)
            files_list = [os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.jpg')]
            file_path = label_folder + self.sys_split + str(len(files_list)) + '.jpg'
            ImgIO.write_img(image, file_path)

    def get_same_label_image_list_byIO(self):
        folders = [os.path.join(self.dataset_name, f) for f in os.listdir(self.dataset_name)
                   if os.path.isdir(os.path.join(self.dataset_name, f))]
        folders = sorted(folders, key=lambda x: x[-1])
        img_label_list = []
        for folder in folders:
            label = folder.split(self.sys_split)[-1]
            file_list = [os.path.join(folder, f) for f in os.listdir(folder)
                         if os.path.isfile(os.path.join(folder, f))]
            img_list = [ImgIO.read_img_uc(path) for path in file_list]
            img_label_list.append((label, img_list))
        return img_label_list

    def get_same_label_image_list(self, char_images, labels):
        img_label_dict = {}.fromkeys(labels)
        for label in img_label_dict.keys():
            img_label_dict[label] = []
        for image, label in zip(char_images, labels):
            img_label_dict[label].append(image)
        return img_label_dict.items()

    def get_label_average_image(self, img_label_list):
        labels = []
        average_imgs = []
        for label, image_list in img_label_list:
            labels.append(label)
            aimg = process.filter_average_image(image_list)
            average_imgs.append(aimg)
        return average_imgs, labels

    def get_label_K_medoids_image(self, img_label_list):
        metric_func = lambda x, y: ShapeContextDistance(x, y, self.character_shape[1], self.character_shape[0], self.sample_number)
        labels = []
        medoids = []
        for label, image_list in img_label_list:
            labels.append(label)
            conv_images = [np_array_to_list(img) for img in image_list]
            mat_cost = spa_dist.squareform(spa_dist.pdist(conv_images, metric=metric_func))
            index = mat_cost.sum(axis=1).argmin()
            print (label, index)
            medoids.append(image_list[index])
        return medoids, labels

    def calculate_fast_prune_keys(self, fast_pruner, characters, r_paras=0.3, threshold=1.00, cut_off=0.78, length=7):
        start_time = time.clock()
        to_train_set = []
        for char_img in characters:
            tmp_list = self.__get_fast_prune_tag(fast_pruner, char_img, r_paras=r_paras, threshold=threshold, cut_off=cut_off)
            if len(tmp_list) <= length and len(tmp_list) > 1 and not tmp_list in to_train_set:
                to_train_set.append(tmp_list)
        finish_time = time.clock()
        print "It takes %.4f min to get fast prune keys." % ((finish_time - start_time) / 60.0)
        return to_train_set

    def get_fast_prune_engines(self, characters, labels, keys):
        fast_prune_engines = {}
        start_digit_time = time.clock()
        image_dict = {}.fromkeys(labels)
        for image, label in zip(characters, labels):
            if not image_dict[label]:
                image_dict[label] = [(image, label)]
            else:
                image_dict[label].append((image, label))
        finish_digit_time = time.clock()
        print "It takes %.4f s to form a character dictionary." % (finish_digit_time - start_digit_time)

        for k in keys:
            start_time = time.clock()
            training = []
            for i in k:
                training.extend(image_dict[i])
            training_images, training_labels = zip(* training)
            training_images = [np_array_to_list(c) for c in training_images]
            tag = ''.join(k)
            fast_prune_engines[tag] = knn_engine(shape=self.character_shape, sample_num=self.sample_number).fit(training_images, training_labels)
            finish_time = time.clock()
            print "tag:" + tag + '\t' + "It takes %.4f min to train a fast prune engine." % ((finish_time - start_time) / 60.0)
        return fast_prune_engines

    def derive_fast_key_engine(self, tag):
        training = []
        for label in tag:
            digits_folder = self.dataset_name + self.sys_split + label
            for digit_file in dataset._get_jpg_list(digits_folder):
                image = ImgIO.read_img_uc(digit_file)
                training.append((image, label))
        training_images, training_labels = zip(* training)
        training_images = [np_array_to_list(c) for c in training_images]
        new_engine = knn_engine(shape=self.character_shape, sample_num=self.sample_number).fit(training_images, training_labels)

        return new_engine

    def __get_fast_prune_tag(self, fast_pruner, image, r_paras=0.3, threshold=1.00, cut_off=0.78):
        img_rsc = GeneralizedShapeContext(image, sample="rsc", sample_params=r_paras)
        tmp_list, tmp_sorted = fast_pruner.get_voting_result(img_rsc, threshold, cut_off)
        return tmp_list
