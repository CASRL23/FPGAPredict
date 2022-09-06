from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans


class EqualGroupsKFold:

    '''
    Another version of KFold.
    This class works exactly as KFold but it puts an equal number of benchmarks from each cluster (or group) into each fold.
    '''
    def __init__(self, n_splits=5, shuffle=False, random_state=None, n_clusters=2, features=None,
                 min_samples=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.features = features # features used to cluster
        self.min_samples = min_samples

    def create_groups(self, x):

        if self.features is None:
            x_copy = np.copy(x)
        else:
            if isinstance(x, pd.DataFrame):
                x_copy = x[self.features]
            else:
                x = np.array(x)
                x_copy = x[:, self.features]
        x_copy = StandardScaler().fit_transform(x_copy)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(x_copy)

        labels = kmeans.labels_

        return labels, self.n_clusters

    def split(self, _x, y=None):
        if isinstance(_x, pd.DataFrame):
            x = _x.copy()
        else:
            x = np.array(_x)
        labels, self.n_clusters = self.create_groups(x)
        if isinstance(x, pd.DataFrame):
            x = x.values.tolist()
        else:
            x = list(x)

        indices = [i for i in range(len(x))]
        folds = [[[], []] for i in range(self.n_splits)]
        f = 0
        for w in range(self.n_clusters):

            # get members of group of label w
            group_indices = np.array(np.where(labels == w))[0]

            n_splits = len(group_indices)

            # group is ready to process
            if self.min_samples is not None:
                if n_splits < self.min_samples:
                    raise ValueError
            if n_splits == 1:
                raise ValueError("KFold: n_splits must be greater than 1. found n_splits={}".format(n_splits))


            kfold = KFold(n_splits=n_splits, shuffle=self.shuffle, random_state=self.random_state)

            for train0, test0 in kfold.split(group_indices):
                test_indices = [group_indices[t] for t in test0]
                folds[f][1] = folds[f][1] + test_indices
                f = f + 1
                if f >= self.n_splits:
                    f = 0
        for f in range(self.n_splits):
            for i in indices:
                try:
                    index = folds[f][1].index(i)
                except ValueError:
                    index = -1
                if index == -1:
                    folds[f][0].append(i)

        return folds


