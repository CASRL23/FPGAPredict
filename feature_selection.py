'''Feature Selection using Exhaustive Search and PCC
'''
import multiprocessing as mp
import time
import random
from itertools import combinations
from datetime import datetime
from estimators import *
from utils import name_file


class ExhaustiveSearch:

    def __init__(self, estimator, n_splits=5, n=15, min=1, max=None, clustering=False,
                 scaling=1, n_cpu=None, parallel=True, printt=True, nruns_error=10):
        self.estimator = estimator  # ML estimator that will be used
        self.printt = printt
        if n_splits <= 1:
            self.whole_dataset = True
            print("Exhaustive Search: selection will be performed on the whole dataset")
        else:
            self.whole_dataset = False

        if max is None:
            self.max = n
        else:
            if max > n:
                self.max = n
            else:
                self.max = max
        self.parallel = parallel  # whether to perform parallel processing or not
        self.n_cpu = n_cpu  # number of cpus to use in parallel, only used if parallel = True
        self.scaling = scaling  # whether to perform scaling or not.
        self.n_splits = n_splits  # int, number of splits with cross validation
        self.n = n  # int, number of features to pass to permutation
        self.min = min  # int, number of minimum features to choose, default is 1
        self.clustering = clustering  # whether permutation is for clustering or not, used to select the highest
        self.features = None  # array-like, name of original features passed
        self.target = None  # array-like, name of target features
        self.benchmarks = None  # array-like, name of data points or benchmarks passed
        self.subset = None  # numpy array, indices of the best features subset of the original features
        self.nruns_error = nruns_error  # if avg error of 1 kfold is below this error, we do 50 runs
        self.nruns_pcc = -1

    # passing x and y as pandas dataframes
    def fit(self, x_to_select_, y_target_, x_prediction_=None):

        if __name__ == "feature_selection":

            # select the highest n PCC ranked features to pass to exhaustive search
            pcc_all = PCC(k=self.n)
                
            x_to_select = pcc_all.fit_transform(x_to_select_, y_target_)
            self.pcc = pcc_all  # save the PCC transformer

            # create a copy of the target feature
            y_target = y_target_.copy()

            # x_clustering is the clustering features if we are performing clustering permutation. Otherwise x_clustering is None
            if x_prediction_ is not None:
                x_clustering = x_prediction_.copy()
            else:
                x_clustering = None

            # print details about the selected PCC features
            if self.printt:
                print("Exhaustive Search: Passed features to permutation are %s\n" % x_to_select.columns.tolist())
                print("PCC values are %s" % pcc_all.best_pcc)

            # determine the type of exhaustive search.
            if self.clustering:
                type_f = "clustering"
            else:
                type_f = "prediction"

            # prepare the files names to store the best permutation results in a file
            if self.printt:
                _name = "perm_results " + str(y_target_.columns[0]) + "_" + str(type_f) + "_" + str(
                    self.estimator.__class__.__name__)
                # verify that this name isn't taken.
                # If taken, add a number next to the name.
                _name = name_file(_name, str(y_target_.columns[0]) + "/", ".xlsx")

                if self.clustering:
                    _name_2 = "best_c_" + str(y_target_.columns[0]) + "_c" + str(self.estimator.get_params()["n_clusters"])
                    _name_2 = name_file(_name_2, str(y_target_.columns[0]) + "/", ".xlsx")

                print("perm features file is stored as %s" % _name)

            # loading data and saving features/benchmarks labels
            self.features = np.array(x_to_select.columns)
            self.target = y_target_.columns
            self.benchmarks = x_to_select.index

            # print to the user what will be running.
            if self.printt:
                print(
                    "Selecting the best " + str(type_f) +
                    " features by using Exhaustive Search.")

            # initialize variables

            min_subset = None
            min_error = np.Inf
            max_pcc = -np.Inf
            count = 0
            same_subset = 0
            prev_subset = None
            finish = False
            choose_list = None
            choose = self.min  # number of features to start with (default is 1 for example)

            # determine up to how many features to try
            upperLimit = self.max

            # prepare a file to store all feature subsets that give an error < 5%
            top_5_subsets = None
            best_c_subsets = None
            all_found_features = []
            found_c_subsets = []

            thr = 48 * 60 * 60  # if elapsed time is more than 48 hours, stop permutation and output results
            # record the start and elapsed time
            start = time.time()
            elapsed = 0

            # determine the number of parallel processes to perform. Only used if self.parallel is True
            if self.parallel and (self.n_cpu is None or not (isinstance(self.n_cpu, int))):
                pnumber = mp.cpu_count() - 1
            elif self.parallel:
                pnumber = self.n_cpu

            while choose <= upperLimit and elapsed < thr and not finish:

                # prepare for parallel processing
                if self.parallel:
                    pool = mp.Pool(pnumber)

                # find the different possible combinations of features for this iteration
                subsets = combinations(self.features, choose)
                processes = []

                i = 0
                # try every subset is subsets
                for subset in subsets:
                    i += 1
                    subset = np.array(subset)

                    # isolate only the features pointed by the variable subset
                    new_x = x_to_select[subset]

                    # check whether to do exhaustive search on whole dataset or with KFold
                    if not self.whole_dataset:
                        if self.parallel:
                            # start an asynchronous parallel process
                            process = pool.apply_async(permutation_fit, args=(
                                new_x, y_target, self.estimator, subset, self.n_splits, self.clustering,

                                self.scaling, x_clustering, self.nruns_error, self.nruns_pcc))
                        else:
                            # call the function directly if without parallel processing
                            process = permutation_fit(new_x, y_target, self.estimator, subset, self.n_splits,
                                                      self.clustering,

                                                      self.scaling, x_clustering, self.nruns_error, self.nruns_pcc)

                    else:  # if on the whole dataset use the function training_fit() instead
                        if self.parallel:
                            process = pool.apply_async(training_fit, args=(new_x, y_target, self.estimator, subset))
                        else:
                            process = training_fit(new_x, y_target, self.estimator, subset)
                    processes.append(process)

                    # wait and extract the results
                    if i >= 100000:
                        # wait for all processes to finish
                        if self.parallel:
                            pool.close()
                            pool.join()

                        for process in processes:
                            if self.parallel:
                                retval = process.get()
                            else:
                                retval = process

                            error = retval[0]
                            subset = retval[1]
                            std_error = retval[2]
                            avg_pcc = retval[3]

                            if error < min_error or (error == min_error and avg_pcc > max_pcc and self.clustering):
                                min_error = error
                                min_subset = subset
                                max_pcc = avg_pcc
                                # do 50 runs in the next iteration only if the error is less than the maximum error
                                if retval[4] < self.nruns_error:
                                    self.nruns_error = retval[4]

                            temp_subset = list(subset)
                            # store the clustering subsets that give 0% error
                            if self.clustering and error == 0 and not (temp_subset in found_c_subsets):
                                found_c_subsets.append(temp_subset)
                                temp = pd.DataFrame([[subset, avg_pcc]],
                                                    columns=["Clustering Features", "Avg. PCC"])
                                if best_c_subsets is None:
                                    best_c_subsets = temp.copy()
                                else:
                                    best_c_subsets = pd.concat([best_c_subsets, temp], ignore_index=True)

                                if len(best_c_subsets.index.tolist()) > 10:
                                    self.nruns_pcc = best_c_subsets["Avg. PCC"].min()

                            # store the top 5 prediction subsets IN ORDER

                            elif not self.clustering and self.printt:

                                transformed_features = list(subset)
                                if (count < 5 or top_5_subsets.iloc[4]["Avg Error (%)"] > error) and not \
                                        (transformed_features in all_found_features):
                                    temp = pd.DataFrame([[transformed_features, error, std_error]],
                                                        columns=["Features", "Avg Error (%)", "STD Error"])
                                    if top_5_subsets is None:
                                        top_5_subsets = temp
                                    else:
                                        top_5_subsets = pd.concat([top_5_subsets, temp], ignore_index=True)
                                    all_found_features.append(transformed_features)
                                    count += 1
                                elif count > 5:
                                    top_5_subsets = top_5_subsets.sort_values("Avg Error (%)", ignore_index=True)
                                    if top_5_subsets.iloc[4]["Avg Error (%)"] > error and not (
                                            transformed_features in all_found_features):
                                        top_5_subsets.drop([4], axis=0, inplace=True)
                                        temp = pd.DataFrame([[transformed_features, error, std_error]],
                                                            columns=["Features", "Avg Error (%)", "STD Error"])
                                        top_5_subsets = pd.concat([top_5_subsets, temp], ignore_index=True)
                                        all_found_features.append(transformed_features)
                        # reset i and the processes
                        i = 0
                        processes = []
                        if self.parallel:
                            pool = mp.Pool(pnumber)

                # wait for all processes to finish
                if self.parallel:
                    pool.close()
                    pool.join()

                for process in processes:
                    if self.parallel:
                        retval = process.get()
                    else:
                        retval = process

                    error = retval[0]
                    subset = retval[1]
                    std_error = retval[2]
                    avg_pcc = retval[3]

                    if error < min_error or (error == min_error and avg_pcc > max_pcc and self.clustering):
                        min_error = error
                        min_subset = subset
                        max_pcc = avg_pcc
                        # do 50 runs in the next iteration only if the error is less than the maximum error
                        if retval[4] < self.nruns_error:
                            self.nruns_error = retval[4]

                    temp_subset = list(subset)
                    # store the clustering subsets that give 0% error
                    if self.clustering and error == 0 and not (temp_subset in found_c_subsets):
                        found_c_subsets.append(temp_subset)
                        temp = pd.DataFrame([[subset, avg_pcc]],
                                            columns=["Clustering Features", "Avg. PCC"])
                        if best_c_subsets is None:
                            best_c_subsets = temp.copy()
                        else:
                            best_c_subsets = pd.concat([best_c_subsets, temp], ignore_index=True)

                        if len(best_c_subsets.index.tolist()) > 10:
                            self.nruns_pcc = best_c_subsets["Avg. PCC"].min()

                    # store the top 5 prediction subsets IN ORDER

                    elif not self.clustering and self.printt:

                        transformed_features = list(subset)
                        if (count < 5 or top_5_subsets.iloc[4]["Avg Error (%)"] > error) and not \
                                (transformed_features in all_found_features):
                            temp = pd.DataFrame([[transformed_features, error, std_error]],
                                                columns=["Features", "Avg Error (%)", "STD Error"])
                            if top_5_subsets is None:
                                top_5_subsets = temp
                            else:
                                top_5_subsets = pd.concat([top_5_subsets, temp], ignore_index=True)
                            all_found_features.append(transformed_features)
                            count += 1
                        elif count > 5:
                            top_5_subsets = top_5_subsets.sort_values("Avg Error (%)", ignore_index=True)
                            if top_5_subsets.iloc[4]["Avg Error (%)"] > error and not (
                                    transformed_features in all_found_features):
                                top_5_subsets.drop([4], axis=0, inplace=True)
                                temp = pd.DataFrame([[transformed_features, error, std_error]],
                                                    columns=["Features", "Avg Error (%)", "STD Error"])
                                top_5_subsets = pd.concat([top_5_subsets, temp], ignore_index=True)
                                all_found_features.append(transformed_features)

                if self.printt:
                    now = datetime.now()
                    # dd/mm/YY H:M:S
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    xcells_subset = min_subset
                    print("Exhaustive Search - Update @ %s: Current Best Subset is %s" % (dt_string, str(
                        xcells_subset)))
                    print("Current min error is %.5f" % (min_error))
                    print("---------")

                elapsed = time.time() - start

                if np.array_equal(min_subset, prev_subset):
                    same_subset += 1
                else:
                    same_subset = 0
                if same_subset >= 3 and same_subset < 5 and choose < self.n - 1:
                    if choose_list is None:
                        choose_list = [h for h in range(choose + 1, self.n + 1)]
                    choose = random.choice(choose_list)
                    choose_list.remove(choose)
                elif same_subset >= 5:
                    finish = True
                else:
                    choose = choose + 1
                prev_subset = np.array(min_subset)
        if top_5_subsets is not None:
            top_5_subsets = top_5_subsets.sort_values("Avg Error (%)")
            excel = pd.ExcelWriter(_name, engine="openpyxl")
            top_5_subsets.to_excel(excel)
            excel.save()

        if self.clustering:
            if best_c_subsets is None:
                best_c_subsets = pd.DataFrame([[min_subset, -1]],
                                              columns=["Clustering Features", "Avg. PCC"])
            # sort the best clustering features based on PCC
            best_c_subsets = best_c_subsets.sort_values("Avg. PCC", ascending=False)

            if self.printt:
                # store them to excel
                excel = pd.ExcelWriter(_name_2, engine="openpyxl")
                best_c_subsets.to_excel(excel)
                excel.save()

            # remove duplicate clusters
            best_c_subsets = best_c_subsets.drop_duplicates(subset=["Avg. PCC"])

            min_subset = best_c_subsets.iloc[0]["Clustering Features"]
            print("Best clustering subsets with 0% clustering error:")
            print(best_c_subsets)

        self.best_c_subsets = best_c_subsets
        self.min_error = min_error
        self.subset = np.array(min_subset)
        return self.subset

    def get_min_error(self):
        return self.min_error

    # transforms features to the new selected feature space
    def transform(self, x_):
        _x = self.pcc.transform(x_)
        new_x = _x[self.subset]

        return new_x

    # fits and transforms features to the new feature space
    def fit_transform(self, x, y, x2=None):
        self.fit(x, y, x2)
        return self.transform(x)

    # returns a boolean mask of the selected features
    def get_support(self):
        support = [False for i in range(len(self.features))]
        for col in range(len(self.features)):
            if self.features[col] in self.subset:
                support[col] = True
            else:
                support[col] = False
        return support


def training_fit(x, y, estimator, subset):
    x = np.array(x)
    y = np.array(y)
    estimator = copy(estimator)
    estimator.fit(x, y)
    training_error = np.divide(np.abs(estimator.predict(x) - y), y) * 100

    return np.mean(training_error), subset, np.std(training_error)


def permutation_fit(x, y, estimator, subset, n_splits=5, clustering=False,
                    scaling=False, x2=None, max_error=10, min_pcc=-1, repeat=50):
    shuffle = True
    random_state = 0
    avg_pcc = -1
    counter = 0
    if clustering:
        # cluster the whole dataset, and record the average PCC among the clusters
        kmeans = copy(estimator)
        new_x = StandardScaler().fit_transform(x)
        kmeans.fit(new_x)
        y_copy = y.copy()  # make a copy of the target feature
        y = kmeans.labels_  # clusters
        # store the benchmarks of each cluster in a dictionary
        n_clusters = kmeans.get_params()['n_clusters']
        original_clusters = {}  # dictionary of the clusters
        benchmarks = np.array(x.index)
        for c in range(n_clusters):
            cluster_indicies = np.array(np.where(y == c))[0]
            original_clusters[c] = cluster_indicies

        y = pd.DataFrame(y)
    # record the error rate of every iteration here
    all_errors = []

    while counter < repeat:

        # increment the random state
        random_state += 1

        if not clustering:

            # to allow few data points (less than 5)
            if len(x) < n_splits:
                n_splits = len(x)
            kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        else:  # use EqualGroupsKFold with clustering
            kfold = EqualGroupsKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state,
                                     n_clusters=estimator.get_params()['n_clusters'], features=None, min_samples=5)

        try:
            splits = kfold.split(x, y)
        except ValueError:  # n_splits
            return 10000, subset, 10000, -1, 10000

        for train0, test0 in splits:

            train_x, test_x = x.iloc[train0], x.iloc[test0]
            train_y, test_y = y.iloc[train0], y.iloc[test0]
            training_benchmarks, testing_benchmarks = np.array(train_x.index), np.array(test_x.index)

            estimator = copy(estimator)

            if clustering or scaling != 0:
                # use standard scaler if the estimator isn't exponential regression and scaling == 1
                if (clustering or scaling == 1) and estimator.__class__.__name__ != "ExponentialRegression":
                    scaler = StandardScaler()

                # use MinMaxScaler if estimator is exponential regression or if scaling isn't 1 or 0
                elif scaling != 1 or estimator.__class__.__name__ == "ExponentialRegression":
                    scaler = MinMaxScaler(feature_range=(10, 11))

                # scale the dataset
                train_x = scaler.fit_transform(train_x)
                test_x = scaler.transform(test_x)

            # fit the estimator. if clustering, this estimator is KMeans.
            estimator.fit(train_x, train_y)

            pred_y = estimator.predict(test_x)

            if clustering:

                training_labels = estimator.labels_

                # an array used to correct the mislabelling
                new_labels = [i for i in range(n_clusters)]
                # check if the clusters are mislabled

                wrong_training_clusters = {}  # dictionary of the clusters that are mispredicted or mislabelled

                not_checked_original_clusters = []  # list of not checked original_cluster
                training_error = 0  # clustering training error
                for c in range(n_clusters):
                    cluster_indicies = np.array(np.where(training_labels == c))[0]
                    cluster_bench = training_benchmarks[cluster_indicies]
                    cluster_error = np.sum(np.isin(cluster_bench, benchmarks[original_clusters[c]], invert=True))
                    if cluster_error == 0:
                        continue
                    elif cluster_error < cluster_bench.shape[0] / 2:
                        training_error += cluster_error
                        continue
                    else:
                        not_checked_original_clusters.append(c)
                        wrong_training_clusters[c] = [cluster_bench, cluster_error]

                for training_cluster in wrong_training_clusters:
                    cluster_bench = wrong_training_clusters[training_cluster][0]
                    min_error = wrong_training_clusters[training_cluster][1]
                    min_c = training_cluster
                    found_better = False
                    for original_cluster in not_checked_original_clusters:
                        cluster_error = np.sum(
                            np.isin(cluster_bench, benchmarks[original_clusters[original_cluster]], invert=True))
                        if cluster_error == 0 or cluster_error < cluster_bench.shape[0] / 2:
                            min_error = cluster_error
                            min_c = original_cluster
                            found_better = True
                            break
                        else:
                            if cluster_error < min_error:
                                min_error = cluster_error
                                min_c = original_cluster
                                found_better = True

                    new_labels[training_cluster] = min_c
                    if found_better:
                        not_checked_original_clusters.remove(min_c)
                    training_error += min_error

                testing_error = 0

                for a in range(len(list(pred_y))):
                    pred_y[a] = new_labels[pred_y[a]]
                test_y = np.array(test_y)
                test_y = test_y.flatten()

                testing_error += np.sum(np.array(pred_y) != test_y)

                error = training_error + testing_error

                all_errors += [error]

            else:
                pred_y = np.array(pred_y)
                pred_y = pred_y.flatten()
                test_y = np.array(test_y)
                test_y = test_y.flatten()
                test_y = np.where(test_y == 0, 0.01, test_y)
                pred_y = np.where(pred_y == 0, 0.01, pred_y)

                error = np.abs(np.divide((pred_y - test_y), test_y) * 100)
                error = list(error)

                all_errors += error
        avg_error = np.mean(all_errors)

        # perform 50 runs if the average error is less than 10%
        if (not clustering and avg_error <= max_error and counter == 0) or counter > 0:
            counter += 1

        elif clustering and avg_error <= max_error and counter == 0:
            # calculate the average pcc
            avg_pcc = 0
            # iterate through each cluster and find the PCC of the top 22 ranked features
            for c in range(n_clusters):
                cluster_indices = original_clusters[c]
                cluster_x = x2.iloc[cluster_indices]
                cluster_y = y_copy.iloc[cluster_indices]
                pcc = PCC(k=22)
                pcc.fit(cluster_x, cluster_y)
                avg_pcc += np.mean(np.abs(pcc.best_pcc))

            avg_pcc = avg_pcc / n_clusters
            if avg_pcc > min_pcc or avg_error < max_error:
                counter += 1
            else:
                counter = repeat # don't do multiple runs
        else:
            counter = repeat # don't do multiple runs

    avg_error = np.mean(all_errors)

    return avg_error, subset, np.std(all_errors), avg_pcc, np.amax(all_errors)


# PCC
'''
    Select using PCC
    k: integer or None, the number of features to select
    threshold: float, if k is None then all features that have PCC less than this number are removed
    '''


class PCC:

    def __init__(self, k=None, threshold=0.75):
        self.k = k
        self.threshold = threshold

    # pass x and y as pandas dataframes
    def fit(self, _x, _y):

        # convert x and y to numpy arrays
        y = np.array(_y)
        x = np.array(_x)
        shape = x.shape[1]
        all_pcc = np.abs(np.corrcoef(x, y, rowvar=False)[-1][:-1])
        if np.isnan(all_pcc).any():
            all_pcc = np.nan_to_num(all_pcc, nan=0)

        if self.k is not None:
            # sort the features based on the PCC values then select the top k features out of them
            mask = all_pcc.argsort()[-self.k:]
        else:
            # if k is not defined by the user, select all features that are bigger than the threshold 
            mask = np.where(all_pcc > self.threshold)

        best_pcc = all_pcc[mask]
        self.mask = mask
        self.best_pcc = best_pcc
        self.shape = shape
        return mask

    def transform(self, _x):
        if isinstance(_x, pd.DataFrame):
            selected_features = np.array(_x.columns)[self.mask]
            new_x = _x[selected_features]
        else:
            x = np.array(_x)
            new_x = x[:, self.mask]
        return new_x

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    # returns a boolean array mask of the selected features
    def get_support(self):
        support = [False for i in range(self.shape)]
        for col in self.mask:
            support[col] = True
        return support

    # returns an integer or columns names mask of the selected features
    def get_support_col(self):
        return self.mask

    def get_highest_pcc(self):
        return self.best_pcc
