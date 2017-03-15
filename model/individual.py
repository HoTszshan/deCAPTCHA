from sklearn import svm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


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
        match = map(lambda x, y: x==y, predicted, expected)
        return len(match) / float(len(y))
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

    def grid_search_fit(self,X, y, C_range=np.logspace(-2, 8, 11), gamma_range = np.logspace(-8, 2, 11), **params):
        print "grid search:"
        param_grid = params['param_grid'] if 'param_grid' in params.keys() else dict(gamma=gamma_range, C=C_range)
        cv  = params['cv'] if 'cv' in params.keys() else StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
        grid = GridSearchCV(self.engine, param_grid=param_grid, cv=cv)
        grid.fit(X, y)
        print("The best parameters are %s with a score of %0.6f" % (grid.best_params_, grid.best_score_))
        print("The parameters of engine is %s" % self.engine.get_params())
        self.engine.set_params(**grid.best_params_)


    def fit(self, X, y, grid_search=False, **params):
        return self.engine.fit(X, y) if not grid_search else self.grid_search_fit(X, y, **params)


class SOMEngine(Engine):

    def __init__(self, kshape=(8, 8), niter=400, distance_metric=None, **params):
        from third_party import SOM
        self.engine = SOM(kshape, niter, **params)
        self.label_maps = {}

    def fit(self, X, y):
        self.engine.train(X)
        for neuron, label in zip(self.engine(X), y):
            self.label_maps[str(neuron)] = label

    def predict(self, X):
        predicted = [self.label_maps[str(i)] for i in self.engine(X)]
        return np.array(predicted)
