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

ESTIMATORS = {0: ClusteringRegression(), 1: LinearRegression(), 2: ExponentialRegression(), 3: RandomForestRegressor(),
              4: MLPRegressor(hidden_layer_sizes=(100,), learning_rate='adaptive', max_iter=500,
                               solver='lbfgs'), 5: DecisionTreeRegressor(),
              6: KNeighborsRegressor(), 7: Ridge()}

ESTIMATORS2 = ESTIMATORS.copy()
for i in range(1,7):
   ESTIMATORS2[i] = ESTIMATORS2[i].__class__.__name__
# creates a copy of an estimator in memory
def copy(estimator):
    try:
        return sklearn.base.clone(estimator)
    except:  # if not an instance of sklearn
        return estimator.copy()
