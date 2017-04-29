# Utils for interface for each stage
# Utils to make pipeline
import os
import sys
import time
import numpy as np
import imgio as ImgIO
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.externals import joblib
from matplotlib import pyplot as plt
# from feature import process2
from tempfile import TemporaryFile


# TODO: all segmentor should have get_characters() function


def save_model(model, model_dir):
    if not os.path.exists(model_dir) and not (len(model_dir.split('.pkl')) >= 2) :
        os.mkdir(model_dir)
    model_name = model_dir if len(model_dir.split('.pkl')) >= 2 else \
       os.path.join(model_dir, 'model.pkl')# os.path.basename(model_dir)+'.pkl')
    joblib.dump(model, model_name)


class ComposeProcessor(object):
    def __init__(self, processors):
        self.__processor = self.__construct_processor(processors)
        self.process_names_ = map(lambda x:getattr(x[0], '__name__'), processors)
        self.__func_params_dict = processors

    def __call__(self, arg, show_process=False):
        if isinstance(arg, str):
            file_path = arg
            with open(file_path) as f:
                image = ImgIO.read_image(f)
        else:
            image = arg
        new_image = self.__processor(image, show_process)
        return new_image

    def __construct_processor(self, processors):
        def processing(image, show_process=False):
            if show_process:
                new_image = image#.copy()
                process_images_=[new_image]
                for func, params in processors:
                    new_image = func(new_image, **params) if func.__code__.co_argcount >= 2 and params else func(new_image)
                    process_images_.append(new_image)
                self.__show_process_flow(process_images_)
                return new_image
            else:
                new_image = image
                for func, params in processors:
                    new_image = func(new_image, **params) if func.__code__.co_argcount >= 2 and params else func(new_image)
                return new_image
        return processing

    def __show_process_flow(self, images):
        image_names = ['original']
        image_names.extend(self.process_names_)
        ImgIO.show_images_list(images, title_names=image_names)

    # For Persistence
    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.__func_params_dict, self.process_names_)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__func_params_dict, self.process_names_ = state
        self.__processor = self.__construct_processor(self.__func_params_dict)


class ComposeExtractor(object):
    def __init__(self, extractors):
        self._extractors = extractors
        #print(extractors)
        #self.extract_names_ = map(lambda x:getattr(x, '__name__'), extractors)

    def __call__(self, arg):
        image_features_dict = {}
        if isinstance(arg, str):
            file_path = arg
            with open(file_path) as f:
                image = ImgIO.read_image(f)
        else:
            image = arg
        for extractor in self._extractors:
            extractor(image, image_features_dict)
        return image_features_dict


class CaptchaDecoder(object):
    def __init__(self, processor, separator, extractor, engine, length=-1, *args, **kwargs):
        self.processor = processor
        self.separator = separator
        self.feature_extractor = extractor
        self.engine = engine
        self.length = length


    def __pre_processing(self, X):
        start_processing = time.time()
        #process_images = map(self.processor, X)
        process_images = [self.processor(image) for image in X]
        finish_processing = time.time()
        print("processing time: %.4f min." % ((finish_processing - start_processing) / 60.0))
        return process_images

    def segment_image(self, image):
        if not self.length == -1:
            try:
                char_images = self.separator(image, length=self.length).get_characters()
                return char_images
            except RuntimeError:
                return None
        else:
            try:
                char_images = self.separator(image).get_characters()
                return char_images
            except RuntimeError:
                return None

    def __segment(self, process_images, y):
        sys.setrecursionlimit(10000)
        characters_features = []
        labels = []
        start_extract = time.time()
        for image, captcha_label in zip(process_images, y):
            if not self.length == -1:
                try:
                    char_images = self.separator(image, length=self.length).get_characters()
                except RuntimeError:
                    ImgIO.show_image(image)
                if len(char_images) == self.length:
                    characters_features.extend(map(self.feature_extractor, char_images))
                    labels.extend(captcha_label)
                else:
                    continue
            else:
                try:
                    char_images = self.separator(image).get_characters()
                except RuntimeError:
                    ImgIO.show_image(image)
                if len(char_images) == len(captcha_label):
                    characters_features.extend(map(self.feature_extractor, char_images))
                    labels.extend(captcha_label)
        finish_extract = time.time()
        print("extract time: %.4f min." % ((finish_extract - start_extract) / 60.0))
        return characters_features, labels

    def fit(self, X, y, **params):
        # start_processing = time.time()
        # #process_images = map(self.processor, X)
        # process_images = [self.processor(image) for image in X]
        # finish_processing = time.time()
        # print("processing time: %.4f min." % ((finish_processing - start_processing) / 60.0))
        #
        # outfile = TemporaryFile()
        # np.savez(outfile, process_images, labels)

        process_images = self.__pre_processing(X)
        characters_features, labels = self.__segment(process_images, y)
        # start_extract = time.time()
        # for image, captcha_label in zip(process_images, y):
        #     char_images = self.separator(image).get_characters()
        #     characters_features.extend(map(self.feature_extractor, char_images))
        #     # ImgIO.show_images_list(char_images)
        #     labels.extend(captcha_label)
        # finish_extract = time.time()
        # # ImgIO.show_image(process_images[0])
        # # for key, value in characters_features[0].items():
        # #     print key, '\t', value

        # Number of training set: 863, extract time: 10.3946 min.
        # start_extract = time.time()
        # images_list = map(lambda x:self.separator(self.processor(x)).get_characters(), X)
        #
        # map(lambda x, y: (characters_features.extend(map(self.feature_extractor,x)), labels.extend(y)), images_list, y)
        # finish_extract = time.time()
        # print("extract time: %.4f min." % ((finish_extract - start_extract) / 60.0))
        # print len(labels), len(characters_features)
        # Normalize
        start_training = time.time()
        self.vectorizer = DictVectorizer()
        train_array = self.vectorizer.fit_transform(characters_features).toarray()
        self.engine.fit(train_array, labels, **params)
        finish_training = time.time()
        print("training time: %.4f min." % ((finish_training - start_training) / 60.0))
        #vprint("parameters of engine is %s" % self.engine.get_params())

    def __make_prediction(self, image, verbose=False):
        start = time.time()
        pre_process_image = self.processor(image)
        # ImgIO.show_image(pre_process_image)
        if not self.length == -1:
            char_images = self.separator(pre_process_image, length=self.length).get_characters()
        else:
            char_images = self.separator(pre_process_image).get_characters()
        # ImgIO.show_images_list(char_images)
        features = map(self.feature_extractor, char_images)
        captcha_features = self.vectorizer.transform(features).toarray()
        t2 = time.time()
        result = self.engine.predict(captcha_features)
        finish = time.time()
        if verbose:
            print finish - start
            # print("It takes %.4f s to predict, the result is %s"%(finish-start, ''.join(result)))
        return ''.join(result)

    def predict(self, X, verbose=False):
        if not hasattr(X, '__iter__'):
            return self.__make_prediction(X, verbose)
        else:
            results = []
            for image in X:
                results.append(self.__make_prediction(image, verbose))
            # TODO: multi thread
            #results = map(self.__make_prediction, x)
            # results.extend('=%')
            return results

    def score(self, X, y, verbose=False):
        pred_labels = self.predict(X, verbose=verbose)
        finish = time.time()
        matches = map(lambda a, b: a==b, y, pred_labels)
        # for e, p in zip(y, pred_labels):
        #     if not e == p:
        #         print e, p
        if verbose:
            expected, predicted = [], []
            map(lambda a, b:(expected.extend(a), predicted.extend(b)), y, pred_labels)
            print("Parameters of the engine is: %s" % self.engine.get_params())
            print("Classification report for classifier %s:\n%s\n" % (self.engine,
                                        metrics.classification_report(expected, predicted, digits=4)))
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
            print("Scores: %.4f" % (float(sum(matches)) / float(len(X))))
        return float(sum(matches)) / float(len(X))

    def get_params(self, *args, **kwargs):
        return self.engine.get_params(*args, **kwargs)

    def save_engine(self, engine_name):
        folder = os.path.join('result', 'engine')
        if not os.path.exists(folder):
            os.mkdir(folder)
        filename = os.path.join(folder, os.path.basename(engine_name))
        joblib.dump(self.engine, filename)


class Profiler(object):

    def __init__(self, model):
        if isinstance(model, str):
            file_path = model
            self.model = joblib.load(file_path)
            self.name = os.path.basename(model).split('.')[0]
        else:
            self.model = model
            self.name = str(model)

    def print_score(self, X, y, verbose=False):
        results = self.model.predict(X)
        matches = map(lambda x, y: x==y, y, results)
        print("The score %s model: %.4f%%" %(self.name.upper(), (float(sum(matches)) / float(len(X)))*100) )
        if verbose:
            for r, o in zip(results, y):
                print("Predict result:" + r + '\t\t\t' + "Label:" + o)
        return (float(sum(matches)) / float(len(X)))

    def predict(self, X):
        return self.model.predict(X)

    def plot_predict_process(self, X):
        process_image = self.model.processor(X)
        segment_images = self.model.segment_image(process_image)
        predicted = list(self.model.predict([X])[0])

        if segment_images:
            plt.figure(num='segmentation')
            plt.gray()
            number = len(segment_images)
            for i in range(number):
                plt.subplot(1, number, i+1)
                plt.imshow(segment_images[i])
                plt.title(predicted[i])
                plt.axis('off')

        plt.figure(num='pre-processing')
        plt.subplot(2, 1, 1)
        plt.title("Image")
        plt.imshow(np.uint8(X), cmap = plt.cm.gray_r)
        plt.axis('off')
        process_image = process_image * 255 if np.max(process_image) <= 1.0 else process_image
        plt.subplot(2, 1, 2)
        plt.gray()
        plt.imshow(process_image)
        plt.title("Processed Image")
        plt.axis('off')


        plt.show()
