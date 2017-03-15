# Utils for interface for each stage
# Utils to make pipeline
import os
import sys
import time
import imgio as ImgIO
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.externals import joblib


# TODO: all segmentor should have get_characters() function


def save_model(model, model_dir):
    if not os.path.exists(model_dir) and not (len(model_dir.split('.pkl')) >= 2) :
        os.mkdir(model_dir)
    model_name = model_dir if len(model_dir.split('.pkl')) >= 2 else \
       os.path.join(model_dir, 'model.pkl')# os.path.basename(model_dir)+'.pkl')
    joblib.dump(model, model_name)


class ComposeProcessor(object):
    def __init__(self, processors, show_process=False):
        self.__processor = self.__construct_processor(processors)
        self.process_names_ = map(lambda x:getattr(x, '__name__'), processors)
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
                for func in processors:
                    new_image = func(new_image)
                    process_images_.append(new_image)
                self.__show_process_flow(process_images_)
                return new_image
            else:
                new_image = image
                for func in processors:
                    new_image = func(new_image)
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
    def __init__(self, processor, separator, extractor, engine, *args, **kwargs):
        self.processor = processor
        self.separator = separator
        self.feature_extractor = extractor
        self.engine = engine

    def fit(self, X, y, **params):
        sys.setrecursionlimit(10000)
        characters_features = []
        labels = []

        start_processing = time.time()
        #process_images = map(self.processor, X)
        process_images = [self.processor(image) for image in X]
        finish_processing = time.time()
        print("processing time: %.4f min." % ((finish_processing - start_processing) / 60.0))
        start_extract = time.time()
        for image, captcha_label in zip(process_images, y):
            char_images = self.separator(image).get_characters()
            characters_features.extend(map(self.feature_extractor, char_images))
            labels.extend(captcha_label)
        finish_extract = time.time()
        print("extract time: %.4f min." % ((finish_extract - start_extract) / 60.0))
        # Number of training set: 863, extract time: 10.3946 min.
        # start_extract = time.time()
        # images_list = map(lambda x:self.separator(self.processor(x)).get_characters(), X)
        #
        # map(lambda x, y: (characters_features.extend(map(self.feature_extractor,x)), labels.extend(y)), images_list, y)
        # finish_extract = time.time()
        # print("extract time: %.4f min." % ((finish_extract - start_extract) / 60.0))

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
        char_images = self.separator(pre_process_image).get_characters()
        features = map(self.feature_extractor, char_images)
        captcha_features = self.vectorizer.transform(features).toarray()
        result = self.engine.predict(captcha_features)
        finish = time.time()
        if verbose:
            print("It takes %.4f s to predict, the result is %s"%(finish-start, ''.join(result)))
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
            return results

    def score(self, X, y, verbose=False):
        pred_labels = self.predict(X)
        matches = map(lambda x, y: x==y, y, pred_labels)
        if verbose:
            expected, predicted = [], []
            map(lambda a, b:(expected.extend(a), predicted.extend(b)), y, pred_labels)
            print("Parameters of the engine is: %s" % self.engine.get_params())
            print("Classification report for classifier %s:\n%s\n" % (self.engine,
                                        metrics.classification_report(expected, predicted)))
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
        return float(sum(matches)) / float(len(X))

    def get_params(self, *args, **kwargs):
        return self.engine.get_params(*args, **kwargs)


class Profiler(object):

    def __init__(self, model):
        if isinstance(model, str):
            file_path = model
            self.model = joblib.load(file_path)
        else:
            self.model = model
        self.name = os.path.basename(model).split('.')[0]

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
