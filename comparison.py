import time

def estimate_function_time(function, n_iter=1, input_list=None):
    start_time = time.time()
    if input_list:
        result = map(lambda n:map(function, input_list), range(n_iter))
    else:
        result = map(function, range(n_iter))
    finish_time = time.time()
    print ("It takes %.4f s to test %s function %d times." % ((finish_time - start_time), function.__name__, n_iter))
    return filter(lambda x:not x == None, result)[0] if len(filter(lambda x:not x == None, result)) > 0 else None




# TODO: grid search to get best parameters
# TODO: size of training set or ratio of testing set
# TODO: cross validation

# TODO: Draw function