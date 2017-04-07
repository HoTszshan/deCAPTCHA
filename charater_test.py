from lib.util import *
from model.individual import *
from feature.simple import *
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from lib import dataset, process
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from model import sc_knn_decoder
import h5py

def test_som(training, testing, feature=None):
    print('Test som...')
    X_train, y_train = training
    X_test, y_test = testing

    x_train = map(feature, X_train)
    x_test = map(feature, X_test)
    vectorizer = DictVectorizer()

    x_train = vectorizer.fit_transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()
    # for i in range(0, len(y_train), 10):
    #     for j in range(i, min(i+10, len(y_train))):
    #         print '%4s' % y_train[j],
    #     print ''

    engine = SOMEngine(kshape=(3,5), niter=4000, learning_rate=0.02)
    engine.fit(x_train, y_train)

    predicted = engine.predict(x_test)
    expected = y_test
    print("Size of training set is: %d, the size of testing set is: %d" % (len(y_train), len(y_test)))
    print("Parameters of the engine is: %s" % engine.get_params())
    print("Classification report for classifier %s:\n%s\n" % (engine,
                                metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))



def test_svm(training, testing, feature, verbose=False):
    print('Test svm...')
    X_train, y_train = training
    X_test, y_test = testing

    vectorizer = DictVectorizer()
    engine = SVMEngine()

    start_training = time.time()
    X_train = map(feature, X_train)
    x_train = vectorizer.fit_transform(X_train).toarray()
    engine.fit(x_train, y_train)
    finish_training = time.time()

    start_testing = time.time()
    X_test = map(feature, X_test)
    x_test = vectorizer.transform(X_test).toarray()
    predicted = engine.predict(x_test)
    finish_testing = time.time()

    expected = y_test
    if verbose:
        print("Parameters of the engine is: %s" % engine.get_params())
        print("Classification report for classifier %s:\n%s\n" % (engine,
                                    metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    return sum(predicted==expected) / float(len(expected)), \
           (finish_training-start_training), (finish_testing-start_testing)/float(len(expected))


def test_knn(training, testing, feature=None, verbose=False):
    X_train, y_train = training
    X_test, y_test = testing
    engine = KNNEngine(k=3)

    # X_train = map(feature, X_train)
    # X_test = map(feature, X_test)
    vectorizer = DictVectorizer()

    # x_train = vectorizer.fit_transform(X_train).toarray()
    # x_test = vectorizer.transform(X_test).toarray()

    start_training = time.time()
    X_train = map(feature, X_train)
    x_train = vectorizer.fit_transform(X_train).toarray()
    engine.fit(x_train, y_train)
    finish_training = time.time()

    start_testing = time.time()
    X_test = map(feature, X_test)
    x_test = vectorizer.transform(X_test).toarray()
    predicted = engine.predict(x_test)
    finish_testing = time.time()

    expected = y_test
    if verbose:
        print("Parameters of the engine is: %s" % engine.get_params())
        print("Classification report for classifier %s:\n%s\n" % (engine,
                                    metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    return sum(predicted==expected) / float(len(expected)), \
           (finish_training-start_training), (finish_testing-start_testing)/float(len(expected))


def test_sc_knn(training, testing, feature=None, verbose=False):
    X_train, y_train = training
    X_test, y_test = testing
    X_train = [process.filter_scale(img, 70, 70) for img in X_train]
    X_test = [process.filter_scale(img, 70, 70) for img in X_test]

    engine = sc_knn_decoder.knn_engine(k=1, shape=(70,70), sample_num=100) #KNNEngine(k=1)
    fast_engine = {}

    def get_same_label_image_list(char_images, labels):
        img_label_dict = {}.fromkeys(labels)
        for label in img_label_dict.keys():
            img_label_dict[label] = []
        for image, label in zip(char_images, labels):
            img_label_dict[label].append(image)
        return img_label_dict.items()
    def get_label_average_image(img_label_list):
        labels = []
        average_imgs = []
        for label, image_list in img_label_list:
            labels.append(label)
            aimg = process.filter_average_image(image_list)
            average_imgs.append(aimg)
        return average_imgs, labels
    def get_label_K_medoids_image(img_label_list):
        metric_func = lambda x, y: sc_knn_decoder.ShapeContextDistance(x, y, 70, 70, 100)
        labels = []
        medoids = []
        for label, image_list in img_label_list:
            labels.append(label)
            conv_images = [sc_knn_decoder.np_array_to_list(img) for img in image_list]
            mat_cost = sc_knn_decoder.spa_dist.squareform(sc_knn_decoder.spa_dist.pdist(conv_images, metric=metric_func))
            index = mat_cost.sum(axis=1).argmin()
            print (label, index)
            medoids.append(image_list[index])
        return medoids, labels
    def __get_fast_prune_tag(fast_pruner, image, r_paras=0.3, threshold=1.00, cut_off=0.78):
        img_rsc = sc_knn_decoder.GeneralizedShapeContext(image, sample="rsc", sample_params=r_paras)
        tmp_list, tmp_sorted = fast_pruner.get_voting_result(img_rsc, threshold, cut_off)
        # ImgIO.show_image(image)
        # print('\n')
        return tmp_list
    def derive_fast_key_engine(tag, char_images, labels):
        to_train_images = []
        img_label_list = get_same_label_image_list(char_images, labels)
        digits_dict = {}
        for label, image_list in img_label_list:
            digits_dict[label] = image_list
        for label in tag:
            # digits_folder = os.path.join(self.dataset_name,label)
            # for digit_file in dataset._get_jpg_list(digits_folder):
            #     image = ImgIO.read_img_uc(digit_file)
            #     training.append((image, label))
            for im in digits_dict[label]:
                to_train_images.append((im, label))
        training_images, training_labels = zip(* to_train_images)
        training_images = [sc_knn_decoder.np_array_to_list(c) for c in training_images]
        new_engine = sc_knn_decoder.knn_engine(k=1, shape=(70, 70), sample_num=100).fit(training_images, training_labels)
        return new_engine

    def figure_known_shapes(char_images, labels):
        img_label_list = get_same_label_image_list(char_images, labels)
        aver_known_images, aver_known_labels = get_label_average_image(img_label_list)
        # kmed_known_images, kmed_known_labels = get_label_K_medoids_image(img_label_list)
        # ImgIO.show_images_list(aver_known_images)

        known_images, known_labels= ([], [])
        aver_known_labels = [label + 'a' for label in aver_known_labels]
        # kmed_known_labels = [label + 'k' for label in kmed_known_labels]
        known_images.extend(aver_known_images)
        known_labels.extend(aver_known_labels)
        known_images_gsc = [sc_knn_decoder.GeneralizedShapeContext(image) for image in known_images]
        return known_images_gsc, known_labels

    def calculate_fast_prune_keys(fast_pruner, characters, r_paras=0.3, threshold=1.00, cut_off=0.78, length=7):
        to_train_set = []
        for char_img in characters:
            tmp_list = __get_fast_prune_tag(fast_pruner, char_img, r_paras=r_paras, threshold=threshold, cut_off=cut_off)
            if len(tmp_list) <= length and len(tmp_list) > 1 and not tmp_list in to_train_set:
                to_train_set.append(tmp_list)
        return to_train_set

    def get_fast_prune_engines(characters, labels, keys):
        fast_prune_engines = {}
        image_dict = {}.fromkeys(labels)
        for image, label in zip(characters, labels):
            if not image_dict[label]:
                image_dict[label] = [(image, label)]
            else:
                image_dict[label].append((image, label))

        for k in keys:
            training = []
            for i in k:
                training.extend(image_dict[i])
            training_images, training_labels = zip(* training)
            training_images = [sc_knn_decoder.np_array_to_list(c) for c in training_images]
            tag = ''.join(k)
            fast_prune_engines[tag] = sc_knn_decoder.knn_engine(k=1,shape=(70,70), sample_num=100).fit(training_images, training_labels)

        return fast_prune_engines

    def __make_fast_prediction(image):
        tag = __get_fast_prune_tag(fast_pruner, image)
        print "Prune Tag:\t" + ''.join(tag)
        if len(tag) == 1:
            return ''.join(tag)
        else:
            # predict
            tag_str = ''.join(tag)
            if tag_str in fast_engine.keys():
                return ''.join(fast_engine[tag_str].predict([sc_knn_decoder.np_array_to_list(image)]))
            else:
                # self.derive_fast_key_engine(tag_str)
                if len(tag_str) <= 3:
                    fast_engine[tag_str] = derive_fast_key_engine(tag_str, X_train, y_train)
                    print "Save new fast engine!"
                    return ''.join(fast_engine[tag_str].predict([sc_knn_decoder.np_array_to_list(image)]))

                else:
                    return ''.join(engine.predict([sc_knn_decoder.np_array_to_list(image)]))

    def fast_predict(x):
        if not hasattr(x, '__iter__'):
            return __make_fast_prediction(x)
        else:
            result = [__make_fast_prediction(image) for image in x]
            return result

    # vectorizer = DictVectorizer()

    start_training = time.time()
    # X_train = map(feature, X_train)
    # x_train = vectorizer.fit_transform(X_train).toarray()
    # engine.fit(x_train, y_train)
    known_gsc, known_labels = figure_known_shapes(X_train, y_train)
    fast_pruner = sc_knn_decoder.FastPruner(known_labels, known_gsc)
    key_list = calculate_fast_prune_keys(fast_pruner, X_train)
    fast_engine = get_fast_prune_engines(X_train, y_train, key_list)
    x_train = [sc_knn_decoder.np_array_to_list(c) for c in X_train]
    engine.fit(x_train, y_train)
    finish_training = time.time()

    start_testing = time.time()
    # X_test = map(feature, X_test)
    # x_test = vectorizer.transform(X_test).toarray()
    # predicted = engine.predict(x_test)
    predicted = fast_predict(X_test)
    finish_testing = time.time()

    expected = y_test
    if verbose:
        print("Parameters of the engine is: %s" % engine.get_params())
        print("Classification report for classifier %s:\n%s\n" % (engine,
                                    metrics.classification_report(expected, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    return sum(predicted==expected) / float(len(expected)), \
           (finish_training-start_training), (finish_testing-start_testing)/float(len(expected))

name = 'gdgs'
new_data_folder = os.path.join('result', 'characters', name)
# X_data, y_data = joblib.load(os.path.join('result', 'recognition', name, 'character.pkl'))
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
# extractor = ComposeExtractor([ScaleExtract(coarse_mesh_count)])
# test_som((X_train, y_train), (X_test, y_test), extractor)


# Save data
# data_file = h5py.File(name+'.h5', 'w')
# data_file.create_dataset('X_train', data=X_train)
# data_file.create_dataset('X_test', data=X_test)
# data_file.create_dataset('y_train', data=y_train)
# data_file.create_dataset('y_test', data=y_test)

def stratified_shuffle_split_data(X, y, test_size):
    for training, testing in StratifiedShuffleSplit(n_splits=1, test_size=test_size).split(X, y):
        X_train = [X[index] for index in training]
        X_test = [X[index] for index in testing]
        y_train = [y[index] for index in training]
        y_test = [y[index] for index in testing]
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def make_more_data():
    if not os.path.exists(new_data_folder):
        os.mkdir(new_data_folder)
    X_data, y_data = joblib.load(os.path.join('result', 'recognition', name, 'character.pkl'))
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    for image, label in zip(X_data, y_data):
        digit_folder = os.path.join(new_data_folder, label)
        if not os.path.exists(digit_folder):
            os.mkdir(digit_folder)
        r = random.random()
        print r
        # img1 = process2.affine_image(image, angle=r*360)
        image = process.filter_scale(image, 32, 32)
        img = process2.rotate_distort(image, factor=r*5)
        # ImgIO.show_images_list([image, img])
        ImgIO.write_image(image, dataset.get_save_image_name(digit_folder,'ori_'+label, img_type='jpg'))
        ImgIO.write_image(img, dataset.get_save_image_name(digit_folder, 'dst_'+label, img_type='jpg'))


    folder_list = [os.path.join(new_data_folder, f) for f in os.listdir(new_data_folder) if not f.endswith('_Store')]

    X_data, y_data = [], []

    for folder in folder_list:
        label = os.path.basename(folder)
        file_list = dataset.get_image_files(folder)
        # while len(file_list) < 1000:
        #     choice = random.choice(file_list)
        #     cho_img = ImgIO.read_image(choice)
        #     r = random.random()
        #     img = process2.rotate_distort(cho_img, factor=r*5)
        #     ImgIO.write_image(img, dataset.get_save_image_name(folder, 'new_'+label, img_type='jpg'))
        #     file_list = dataset.get_image_files(folder)
        # print file_list
        # print len(file_list)
        images = map(ImgIO.read_image, file_list)
        for image in images:
            X_data.append(image)
            y_data.append(label)

    joblib.dump((X_data, y_data), "character.pkl")

def mean_size_dataset_evaluate(X, y, training_ratio, classifier, extractor, n_times=1):
    sss = StratifiedShuffleSplit(n_splits=n_times, test_size=1-training_ratio)
    x = map(extractor, X)
    vectorizer = DictVectorizer()
    x = vectorizer.fit_transform(x).toarray()
    scores = cross_val_score(classifier, x, y, cv=sss)
    print('################################################################')
    print("training set: %.4f" % training_ratio)
    print("classifier: %s" % classifier)
    print("features: %s" % extractor)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print('################################################################')

X_data, y_data  = joblib.load('character.pkl')

classifier = KNNEngine(k=3)
extractor = ComposeExtractor([coarse_mesh_count])#[ScaleExtract(position_brightness)])
# mean_size_dataset_evaluate(X_data, y_data, 0.01, classifier, extractor, n_times=5)

def spilt_acc_time(X, y, training_ratio, evaluator, extractor, n_times):
    scores = []
    training_times, testing_times = [], []
    for i in range(n_times):
        X_train, X_test, y_train, y_test = stratified_shuffle_split_data(X, y, test_size=1-training_ratio)
        score, training_t, testing_t = evaluator((X_train, y_train), (X_test, y_test), extractor)
        scores.append(score)
        training_times.append(training_t)
        testing_times.append(testing_t)
    scores = np.array(scores)
    training_times = np.array(training_times)
    testing_times = np.array(testing_times)
    print('################################################################')
    print("training set: %.4f" % training_ratio)
    print("classifier: %s" % classifier)
    print("features: %s" % extractor)
    print("Scores: %s" % str(scores))
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print("Training time: %s" % str(training_times))
    print("Training time: %0.4fs (Max: %.4fs,  Min: %.4fs)" % (training_times.mean(), training_times.max(), training_times.min()))
    print("Testing time: %s" % str(testing_times))
    print("Testing time: %0.4fs (Max: %.4fs,  Min: %.4fs)" % (testing_times.mean(), testing_times.max(), testing_times.min()))
    print('################################################################')
    print('\n\n\n')


ratio = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
         0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
# spilt_acc_time(X_data, y_data, 0.01, test_knn, extractor, n_times=5)
map(lambda x: spilt_acc_time(X_data, y_data, x, test_svm, extractor, n_times=5), ratio)

