'''
This file includes all estimators used in this project.
'''

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from exponential_regression import ExponentialRegression
from sklearn.model_selection import KFold
import sklearn
from sklearn.cluster import KMeans
from clustering_regression import ClusteringRegression
from kfold import *

ESTIMATORS = {1: LinearRegression(), 2: ExponentialRegression(), 3: RandomForestRegressor(),
              4: MLPRegressor(hidden_layer_sizes=(100,), learning_rate='adaptive', max_iter=500,
                               solver='lbfgs'), 5: DecisionTreeRegressor(),
              6: KNeighborsRegressor(), 7: Ridge(),
              "Linear": LinearRegression(), "Exp": ExponentialRegression(), "RF": RandomForestRegressor(),
              "ANN": MLPRegressor(hidden_layer_sizes=(100,), learning_rate='adaptive', max_iter=500,
                               solver='lbfgs'),
              "DecTree":DecisionTreeRegressor(),
              "KNN":KNeighborsRegressor(),
              "Ridge":Ridge()
              }
# creates a copy of an estimator in memory
def copy(estimator):
    try:
        return sklearn.base.clone(estimator)
    except:  # if not an instance of sklearn
        return estimator.copy()
