from sklearn.linear_model import LinearRegression
import sklearn
import pandas as pd
import numpy as np


class ExponentialRegression:
    '''
    Exponential Regression: A class written based on the built-in function of Linear Regression
        estimator: the estimator to use, LinearRegression is recommended. Other estimators may not work correctly.

    '''
    def __init__(self, estimator=LinearRegression()):
        self.estimator = estimator

        self.n_features_in_ = 0
        self.fitted = None
        self.intercept_ = None
        self.coef_ = None

    '''
    set the parameters.
    '''
    def set_parameters(self, n_features_in_, fitted):
        self.estimator = LinearRegression()
        self.n_features_in_ = n_features_in_
        self.fitted = fitted

    '''
    fit the estimator on the training set.
    '''
    def fit(self, _x, _y):
        if isinstance(_x, pd.DataFrame):
            x = _x.values.tolist()
        else:
            x = _x
        if isinstance(_y, pd.DataFrame):
            y = _y.values.tolist()
        else:
            y = _y
        x = np.array(x)
        y = np.array(y)

        x = np.log(x)
        y = np.log(y)

        mask = np.isinf(x)
        mask2 = np.isnan(x)
        x = np.where(mask == True, 1, x)
        x = np.where(mask2 == True, 1, x)

        mask = np.isinf(y)
        mask2 = np.isnan(y)
        y = np.where(mask == True, 1, y)
        y = np.where(mask2 == True, 1, y)
        self.n_features_in_ = x.shape[1]

        regr = sklearn.base.clone(self.estimator)

        x = x.tolist()
        y = y.flatten().tolist()
        self.fitted = regr.fit(x, y)  # fit the data

        ret_regr = ExponentialRegression()
        ret_regr.set_parameters(self.n_features_in_, self.fitted)

        return ret_regr


    '''
    predict a new benchmark
    '''
    def predict(self, _x):

        x = np.array(_x)

        x = np.log(x)
        mask = np.isinf(x)
        mask2 = np.isnan(x)
        x = np.where(mask == True, 1, x)
        x = np.where(mask2 == True, 1, x)

        regr = self.fitted
        a = regr.intercept_
        a = [float(a)]
        b = regr.coef_
        self.intercept_ = a
        self.coef_ = b
        pred_arr = []
        
        for i in range(x.shape[0]):
            dot = np.dot(np.transpose(b), x[i])
            pred = np.exp(a) * np.exp(dot)
            pred_arr.append(pred)

        mask1 = np.isnan(pred_arr)
        pred_arr = np.where(mask1,100, pred_arr)
        mask2 =  np.isinf(pred_arr)
        pred_arr = np.where(mask2, 100, pred_arr)
        return np.array(pred_arr)

    '''
    reset the estimator only (create a copy in memory that is not fitted)
    '''
    def reset(self):
        self.estimator = copy(self.estimator)

    '''
    Create a copy of this class in memory
    '''
    def copy(self):
        return ExponentialRegression(estimator=copy(self.estimator))

    '''
    string representation of this class
    '''
    def __repr__(self):
        return "ExponentialRegression()"


# creates a copy of an estimator in memory
def copy(estimator):
    try:
        return sklearn.base.clone(estimator)
    except:  # if not an instance of sklearn
        return estimator.copy()