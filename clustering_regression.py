from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ClusteringRegression:
    '''
        Clustering Regression Initialization Function.
        estimators: A 2D array of the estimator and features to use
                    index 0 corresponds to the estimator and features for cluster 0
                    index i corresponds to the estimator and features for cluster i, and so on.
                    at every index i, there is an array that has the estimator followed by the features to use in the second position.
                    i.e. [[estimator for c1, features for c1], ..... [estimator for ci, features for ci]]
        clustering_features: the name of the features used to cluster the data set.
        original_clusters: a dictionary of the original clusters with the benchmarks names.
    '''

    def __init__(self, estimators=None, clustering_features=None, original_clusters=None):
        if estimators is not None:
            self.estimators = list(estimators)
            self.n_clusters = len(estimators)
        else:
            self.estimators = None
            self.n_clusters = None
        if clustering_features is not None:
            self.clustering_features = np.copy(clustering_features)
        else:
            self.clustering_features = None
        if original_clusters is not None:
            self.original_clusters = original_clusters
        else:
            self.original_clusters = None
        self.kmeans = None

    '''
    sets the parameter for this estimator.
    estimators: A 2D array of the estimator and features to use
                index 0 corresponds to the estimator and features for cluster 0
                index i corresponds to the estimator and features for cluster i, and so on.
                at every index i, there is an array that has the estimator followed by the features to use in the second position.
                i.e. [[estimator for c1, features for c1], ..... [estimator for ci, features for ci]]
    clustering_features: the name of the features used to cluster the data set.
    original_clusters: a dictionary of the original clusters with the benchmarks names.
    '''

    def set_parameters(self, estimators=None, clustering_features=None, original_clusters=None):

        if estimators is not None:
            for b in range(len(estimators)):
                estimators[b] = list(estimators[b])
            self.estimators = list(estimators)
            self.n_clusters = len(estimators)
        if clustering_features is not None:
            self.clustering_features = np.copy(clustering_features)
        if original_clusters is not None:
            self.original_clusters = original_clusters

        self.kmeans = None

    '''
    creates the cluster and fits the estimator for each cluster.

    x: a pandas dataframe of the input features.
    y: a pandas dataframe of the target feature.
    x2: a pandas dataframe of the clustering features.
    '''

    def fit(self, x, y, x2=None):
        if x2 is None:
            x2 = x
        self.clustering_features = list(np.copy(self.clustering_features))

        # save the training benchmarks name
        training_benchmarks = x.index.tolist()

        # scale the dataset
        scaler = StandardScaler()
        x2 = scaler.fit_transform(x2[self.clustering_features], y)

        # save the scaler
        self.scaler = scaler

        # cluster the dataset
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(x2)
        self.kmeans = kmeans
        labels = kmeans.labels_
        training_labels = list(labels)

        # get the original clusters
        original_clusters = self.original_clusters

        # an array used to correct the mislabelling
        new_labels = [i for i in range(self.n_clusters)]

        # check if the clusters are mislabled
        # training_clusters is a dictionary that has the clusters
        # for example {0: [DFADD, DFMUL], 1: [MIPS, JPEG]}
        training_clusters = {}  # dictionary of the clusters

        for i in range(len(training_labels)):
            try:
                training_clusters[training_labels[i]].append(training_benchmarks[i])
            except KeyError:
                training_clusters[training_labels[i]] = [training_benchmarks[i]]

        # iterate over each cluster and relable the clusters
        for i in range(self.n_clusters):
            clusters_error = []
            for j in range(self.n_clusters):
                cluster_error = count_intersect(original_clusters[j], training_clusters[i])

                clusters_error.append(cluster_error)

            new_label = np.argmin(clusters_error)
            new_labels[i] = new_label

        # fit the estimator of each cluster
        for i in range(self.n_clusters):

            # get the indices of this cluster (i.e. the benchmarks of this cluster)
            cluster_indices = np.array(np.where(labels == i))[0]

            # find the updated label for this cluster
            c = new_labels[i]

            # get the features for this cluster
            features = self.estimators[c][1]

            # get the input features for this cluster
            new_x = x[features]

            # take the input features of only the benchmarks of this cluster
            new_x = new_x.copy().to_numpy()[cluster_indices]

            # take the target metric of only the benchmarks of this cluster
            new_y = y.copy().to_numpy()[cluster_indices]

            if self.estimators[c][0].__class__.__name__ != "ExponentialRegression":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler(feature_range=(10, 11))
            new_x = scaler.fit_transform(new_x, new_y)

            # create a copy of the estimator of this cluster
            self.estimators[c][0] = copy(self.estimators[c][0])

            # fit the estimator for this cluster
            self.estimators[c][0].fit(new_x, new_y)
            self.estimators[c].append(scaler)

        # save the new labels to use when predicting new data
        self.new_labels = new_labels

    '''
    predicts the target metric for new benchmarks. 
    x is a pandas dataframe of the input features for the new benchmarks.
    x2 is another pandas dataframe of the clustering features for the same new benchmarks.

    x and x2 must have the same benchmarks and in the same order.
    '''

    def predict(self, x, x2=None):
        if x2 is None:
            x2 = x
        # scale the clustering features
        x2 = self.scaler.transform(x2[self.clustering_features])

        # predict the clusters of the benchmarks
        clusters = self.kmeans.predict(x2)

        # convert to the new updated labels
        clusters = [self.new_labels[clusters[i]] for i in range(len(clusters))]

        # predict the target metric for each benchmark and save it in pred_arr
        pred_arr = []

        # create an intercept_ and coef_ arrays. Used when estimator is Linear or Exponential regression
        self.intercept_ = []
        self.coef_ = []

        # create features array to show the features used to predict.
        self.features = []

        # create estiamtors array to show the estimators used to predict.
        self.estimators_pred = []

        # iterate through each benchmark
        for row in range(len(clusters)):

            # get the cluster of the benchmark
            cluster = clusters[row]

            # get the features of the benchmark
            features = self.estimators[cluster][1]

            # get input features
            new_x = x[features].copy().to_numpy()

            # get the scaler and scale
            scaler = self.estimators[cluster][2]
            new_x = scaler.transform([new_x[row]])
            # append features and estimator
            self.features.append(features)
            self.estimators_pred.append(self.estimators[cluster][0].__class__.__name__)

            # save the coef_ and intercept_
            if self.estimators[cluster][0].__class__.__name__ == "LinearRegression" or self.estimators[cluster][
                0].__class__.__name__ == "ExponentialRegression":
                self.coef_.append(self.estimators[cluster][0].coef_)
                self.intercept_.append(self.estimators[cluster][0].intercept_)
            else:
                self.coef_ = None
                self.intercept_ = None

            # predict the target metric and append it to the array pred_arr
            pred_arr.append(self.estimators[cluster][0].predict([new_x])[0])

        # save the clusters
        self.clusters = clusters

        return np.array(pred_arr)

    '''
    creates a copy in memory of the estimator with the same parameters
    '''

    def copy(self):
        for i in range(len(self.estimators)):
            self.estimators[i][0] = copy(self.estimators[i][0])

        return ClusteringRegression(estimators=self.estimators, clustering_features=self.clustering_features,
                                    original_clusters=self.original_clusters)

    '''
    returns a string representation of this class.
    '''

    def __repr__(self):
        args = ""
        prev = False
        if self.estimators is not None:
            prev = True
            temp = "estimators="
            for estimator in self.estimators:
                temp += str(estimator)
            args += temp
        if self.clustering_features is not None:
            if prev:
                args += ","
            prev = True
            args += "clustering_features=" + str(self.clustering_features)
        if self.original_clusters is not None:
            if prev:
                args += ","
            prev = True
            args += "original_clusters=" + str(self.original_clusters)
        return "ClusteringRegression({})".format(args)


# creates a copy of an estimator in memory
def copy(estimator):
    try:
        return sklearn.base.clone(estimator)
    except:  # if not an instance of sklearn
        return estimator.copy()


# count how many elements in b are NOT in a.
def count_intersect(a, b):
    error = 0  # how many elements in a are NOT in b
    for element in a:
        if not element in b:
            error += 1
    return error
