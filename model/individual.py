from __future__ import print_function
from sklearn import svm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from third_party import SOM
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import *
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
# from feature import process2
from lib import imgio as ImgIO
# from keras.models import load_model
import h5py
import random

def l2_dis(x, y):
    return (x ** 2 + y ** 2) ** 0.5

class Engine(object):
    def __init__(self, engine):
        self.engine = engine

    def fit(self, X, y):
        return self.engine.fit(X, y)

    def predict(self, X):
        return self.engine.predict(X)

    def score(self, X, y):
        predicted = self.predict(X)
        expected = y
        match = map(lambda a, b: a==b, predicted, expected)
        return sum(match) / float(len(y))
        #return self.engine.score(X, y)

    def get_params(self, *args, **kwargs):
        return self.engine.get_params(*args, **kwargs)

    def set_params(self, *args, **kwargs):
        self.engine.set_params(*args, **kwargs)


class SVMEngine(Engine):

    def __init__(self, **params):
        self.engine = svm.SVC(**params)

    def __call__(self, *args, **kwargs):
        return self.engine

    def grid_search_fit(self,X, y, C_range=np.logspace(-2, 3, 6), gamma_range = np.logspace(-4, 2, 7), **params):
        print("grid search:")
        param_grid = params['param_grid'] if 'param_grid' in params.keys() else dict(gamma=gamma_range, C=C_range)
        cv  = params['cv'] if 'cv' in params.keys() else StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        grid = GridSearchCV(self.engine, param_grid=param_grid, cv=cv)
        grid.fit(X, y)
        print("The best parameters are %s with a score of %0.6f" % (grid.best_params_, grid.best_score_))
        print("The parameters of engine is %s" % self.engine.get_params())
        self.engine.set_params(**grid.best_params_)
        self.engine.fit(X, y)


    def fit(self, X, y, grid_search=False, **params):
        return self.engine.fit(X, y) if not grid_search else self.grid_search_fit(X, y, **params)

class PCA_SVMEngine(Engine):
    def __init__(self, **params):
        self.engine = make_pipeline(PCA(), svm.SVC(**params))


class SOMEngine(Engine):

    def __init__(self, kshape=(8, 8), niter=400, distance_metric=None, **params):
        distance_metric = distance_metric if distance_metric else l2_dis
        self.engine = SOM(kshape, niter, distance_metric=distance_metric, **params)
        self.label_maps = {}
        # self.__train_data = None

    def fit(self, X, y):
        from collections import Counter
        # self.engine.train(new_)
        # for neuron, label in zip(self.engine(new_x), new_y):
        self.engine.train(X)
        for neuron, label in zip(self.engine(X), y):
            if str(neuron) in self.label_maps.keys():
                self.label_maps[str(neuron)].append(label)
            else:
                self.label_maps[str(neuron)] = [label]
        for key, c_labels in self.label_maps.items():
            # print(key, '\t', sorted(c_labels))
            label_counts = Counter(c_labels)
            self.label_maps[key] = label_counts.most_common(1)[0][0]

    def __make_mapping(self, key):
        if str(key) in self.label_maps.keys():
            return self.label_maps[str(key)]
        else:
            x, y = key[0], key[1]
            dire = [(-1, 0), (+1, 0), (1, 0), (-1, 0)]
            near_value = [self.label_maps[str(np.array([x+i, y+j]))] for i, j in dire
                       if str(np.array([x+i, y+j])) in self.label_maps.keys()]
            if near_value:
                return random.choice(near_value)
            else:
                return random.choice(self.label_maps.values())

    def predict(self, X):
        # print(type(self.engine(X)[0]))
        # predicted = [self.label_maps[str(i)] for i in self.engine(X)]
        predicted = [self.__make_mapping(i) for i in self.engine(X)]
        return np.array(predicted)

    def get_params(self, *args, **kwargs):
        params = {}
        params['kshape'] = self.engine.kshape
        params['niter'] = self.engine.niter
        params['learning_rate'] = self.engine.lrate
        params['iradius'] = self.engine.radius
        params['distance_metric'] = self.engine.distance_metric
        # params['initialization_func']
        return params


    # # For Persistence
    # def __getstate__(self):
    #     """Return state values to be pickled."""
    #     params = self.get_params()
    #     return (params, self.__train_data, self.label_maps)
    #
    # def __setstate__(self, state):
    #     """Restore state from the unpickled state values."""
    #     params, self.__train_data, self.label_maps= state
    #     self.engine = SOM(**params)
    #     self.engine.train(self.__train_data)


class KNNEngine(Engine):

    def __init__(self, k=3, **params):
        self.engine = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', n_jobs=-1)

class EasySVMEngine(Engine):

    def __init__(self, **params):
        self.engine = svm.LinearSVC(**params)

# class CNNEngine(Engine):
#
#     def __init__(self, batch_size=32, num_classes=9, epochs=12, img_rows=28, img_cols=28):
#         self.batch_size = batch_size
#         self.num_class = num_classes
#         self.epochs = epochs
#         self.img_rows = img_rows
#         self.img_cols = img_cols
#         if K.image_data_format() == 'channels_first':
#             self.input_shape = (1, img_rows, img_cols)
#         else:
#             self.input_shape = (img_rows, img_cols, 1)
#         self.label_map = {}
#         self.engine = Sequential()
#         self.engine.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=self.input_shape))
#         self.engine.add(Conv2D(64, (3, 3), activation='relu'))
#         self.engine.add(MaxPooling2D(pool_size=(2, 2)))
#         self.engine.add(Dropout(0.25))
#         self.engine.add(Flatten())
#         self.engine.add(Dense(128, activation='relu'))
#         self.engine.add(Dropout(0.5))
#         self.engine.add(Dense(num_classes, activation='softmax'))
#         self.engine.summary()
#         self.engine.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adadelta(),
#                   metrics=['accuracy'])
#
#     def fit(self, X, y):
#         for label, index in zip(np.unique(y),range(self.num_classes)):
#             self.label_map[label] = index
#
#         X_train = np.array([process2.inverse(process2.resize_transform(img,
#                 output_shape=(self.img_rows, self.img_cols))) for img in X])
#
#         if K.image_data_format() == 'channels_first':
#             X_train = X_train.reshape(X_train.shape[0], 1, self.img_rows, self.img_cols)
#         else:
#             X_train = X_train.reshape(X_train.shape[0], self.img_rows, self.img_cols, 1)
#         x_train = X_train.astype('float32')
#         if x_train[0].max() > 1.0:
#             x_train /= 255
#         y_train = np.array([self.label_map[v] for v in y])
#         y_train = keras.utils.to_categorical(y_train, self.num_classes)
#         self.engine.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
#           verbose=1)#, validation_data=(x_test, y_test))
#
#     def predict(self, X):
#         # predect self.engine.predict(X)
#         if not hasattr(X, '__iter__') or X.ndim <=3:
#             X = np.array([X])
#         X_test = np.array([process2.inverse(process2.resize_transform(img,
#                 output_shape=(self.img_rows, self.img_cols))) for img in X])
#         if K.image_data_format() == 'channels_first':
#             x_test = X_test.reshape(X_test.shape[0], 1, self.img_rows, self.img_cols)
#         else:
#             x_test = X_test.reshape(X_test.shape[0], self.img_rows, self.img_cols, 1)
#         if x_test[0].max() > 1.0:
#             x_test /= 255
#         predict = self.engine.predict_classes(x_test, batch_size=1, verbose=1)
#         result = [self.label_map[i] for i in predict]
#         return result


