import numpy as np

from feature_selection import *
from datetime import date
from time import perf_counter
from utils import name_file
import matplotlib.pyplot as plt
import openpyxl


class FPGAPredict:

    def __init__(self, params):
        global ESTIMATORS
        self.estimator = params["estimator"]
        self.parallel = params["parallel"]  # whether to perform parallel processing with exhaustive search
        self.n_cpu = params["n_cpu"]  # number of cores to use with parallel processes
        self.n_runs = params["n_runs"]  # number of random runs to perform
        self.print_bool = params["print"]  # whether to print the results or not
        self.excelname = params["excelname"]  # the name of the excel file that has the data
        self.benchmarks = None  # name of the benchmarks
        self.n_clusters = params["n_clusters"]  # number of clusters to create
        self.n_splits = params["n_splits"]  # number of splits with KFold
        self.scaling = params["scaling"]

    '''
    Description:
    reads the excel file that has the dataset.
    The excel file can be of format .xlsx or .xls and it must have the sheets:
    "features" that has the input features and "target" which has the target metrics
    It may also have "clustering_features" sheet for the clustering features.
    If it does not have the latter sheet, the clustering features will be read from the "features" sheet.
    parameters:
    - features: the names of the input features to read from the excel file. If None, read all the "features" sheet.
    - target: the names of the target features to read from the excel file. If None, read all the "target" sheet. 
        Note: It is strongly recommended to read only one target metric at a time. 
    - clustering: the names of the clustering features to read from the excel file. If None, read all the "features" or "clustering_features" sheet.
    returns: three dataframes of the same index as follows.
        x_input: contains the input features dataset.
        y_target: contains the target metrics dataset.
        x_clustering: contains the clustering features dataset.
    '''

    def read_excel_sheet(self, features, target, clustering):

        # either read only part of the excel file or read it all based on the configurations
        if features is not None:
            features = list(features)
            features.insert(0, "Benchmarks")
        if target is not None:
            target = list(target)
            target.insert(0, "Benchmarks")
        if clustering is not None:
            clustering = list(clustering)
            clustering.insert(0, "Benchmarks")

        # run with xlrd if the file is xls, otherwise run with openpyxl
        if self.excelname.split(".")[1] == "xls":
            engine = "xlrd"
        else:
            engine = "openpyxl"

        x_input = pd.read_excel(self.excelname, sheet_name="features", engine=engine, index_col=0,
                                usecols=features)  # read features sheet
        y_target = pd.read_excel(self.excelname, sheet_name="target", engine=engine, index_col=0,
                                 usecols=target)  # read target sheet
        try:
            x_clustering = pd.read_excel(self.excelname,
                                         sheet_name="clustering_features", engine=engine, index_col=0,
                                         usecols=clustering)  # read clustering features

        except KeyError:
            x_clustering = pd.read_excel(self.excelname,
                                         sheet_name="features", engine=engine, index_col=0,
                                         usecols=clustering)  # read clustering features

        x_input_rows_with_NaN = x_input.isnull().any(axis=1)
        y_target_rows_with_NaN = y_target.isnull().any(axis=1)
        x_clustering_rows_with_NaN = x_clustering.isnull().any(axis=1)
        rows_with_NaN = x_input_rows_with_Nan & y_target_rows_with_NaN & x_clustering_rows_with_NaN
        x_input = x_input[~rows_with_NaN]
        y_input = y_input[~rows_with_NaN]
        x_clustering = x_clustering[~rows_with_NaN]
        NaN_benchmarks = x_input[rows_with_NaN].index.tolist()
        if len(NaN_benchmarks) > 0:
            print(
                "FPGAPredict: The following benchmark(s) were removed from the dataset because they have NaN features: \n %s\n" % (
                    NaN_benchmarks))
        # save features and benchmarks names
        self.features = x_input.columns.tolist()
        self.benchmarks = x_input.index.tolist()
        self.target_name = y_target.columns
        return x_input, y_target, x_clustering

    '''
    Description:
    selects the clustering features using exhaustive search (only if the parameter ex_search=True).
    Exhaustive search internally uses the Pearson Correlation Coefficient (PCC) to select up 
    to n features and then pass them to Exhaustive Search.
    parameters:
        -x_input: a dataframe of the input features dataset.
        -y_target: a dataframe of the target features dataset.
        -x_clustering: a dataframe of the clustering features dataset.
        -ex_search: a boolean, whether to perform exhaustive search or not.
        -min_features: integer, the minimum features to try in exhaustive search.
        -max_features: integer, the maximum features to try in exhaustive search.
        -n: integer, the number of features to select (using PCC) and pass to exhaustive search.

    returns: A 2-D list of the top clustering features.
        -top_clustering_features: a 2-D list containing all top clustering features subsets that gave 0% clustering error 
            or
         only one clustering features subset that has the minimum clustering error (if not 0%)
    '''

    def select_clustering_features(self, x_input, y_target, x_clustering, ex_search=False, min_features=2,
                                   max_features=None, n=10):

        # perform exhaustive search if ex_search=True
        if ex_search:
            print(
                "FPGAPredict: selecting clustering features by passing the top %d features to exhaustive search.." % n)

            # initialize Exhaustive Search using KMeans
            ex_selector = ExhaustiveSearch(estimator=KMeans(n_clusters=self.n_clusters, random_state=42),
                                           n_splits=self.n_splits, n=n, clustering=True,
                                           max=max_features, min=min_features,
                                           parallel=self.parallel, n_cpu=self.n_cpu, printt=self.print_bool)

            # record the start time
            time1 = perf_counter()

            # start exhaustive search to select clustering features by calling fit()
            ex_selector.fit(x_clustering, y_target, x_input)  # transforms clustering features to the top choice

            # record the end time and calculate elapsed time
            time2 = perf_counter()
            elapsed = time2 - time1

            # get the top clustering features from ex_selector
            top_clustering_features = np.array(ex_selector.best_c_subsets["Clustering Features"])

            # print some details
            print("FPGAPredict: The clustering features were selected successfully")
            print(
                "FPGAPredict: ELAPSED TIME FOR SELECTING THE FEATURES IS: %.4f seconds.\nStart Time = %.4f seconds and End Time = %.4f seconds\n" % (
                    elapsed, time1, time2))
        else:
            top_clustering_features = [x_clustering.columns.tolist()]

        return top_clustering_features

    '''
    Description:
    clusters the dataset and selects the prediction features for each cluster separately.

    parameters:
        -x_input: a dataframe of the input features dataset.
        -y_target: a dataframe of the target features dataset.
        -x_clustering: a dataframe of the clustering features dataset.
        -ex_search: a boolean, whether to perform exhaustive search or not.
        -min_features: integer, the minimum features to try in exhaustive search.
        -max_features: integer, the maximum features to try in exhaustive search.
        -n: integer, the number of features to select (using PCC) and pass to exhaustive search.
        -estimator_4each_cluster: A 2D list containing the estimator for each cluster. The length of the list must equal self.n_clusters.

    returns: 2D list with the features for each cluster
        -features_4each_cluster: A 2D list containing the selected features subset for each cluster.
                                 The length of the list equals self.n_clusters
    '''

    def select_prediction_features(self, x_input, y_target, x_clustering, ex_search=False, n=10, min_features=2,
                                   max_features=None, estimator_4each_cluster=None):
        print("FPGAPredict: Selecting the prediction features for each cluster ..")

        if estimator_4each_cluster is None or np.array(estimator_4each_cluster).shape[0] < self.n_clusters:
            estimator_4each_cluster = [self.estimator for k in range(self.n_clusters)]

        # get the clusters (input and target datasets and the benchmarks corresponding to each cluster)
        x_4each_cluster, y_4each_cluster, benchmarks_4each_cluster = self.get_clusters(x_input, y_target, x_clustering)

        # prepare an array of selected features
        selected_features_4each_cluster = []

        # the cluster label, incremented with every loop iteration
        cluster = -1

        for x_cluster, y_cluster in list(zip(x_4each_cluster, y_4each_cluster)):
            cluster += 1

            if self.print_bool:
                print("\nFPGAPredict: Processing cluster %d ...\n" % cluster)

            # perform Exhaustive Search for prediction if ex_search=True
            if ex_search:
                ex_selector = ExhaustiveSearch(estimator=estimator_4each_cluster[cluster], n_splits=self.n_splits,
                                               min=min_features, max=max_features,
                                               printt=self.print_bool,
                                               n=n, scaling=self.scaling,
                                               parallel=self.parallel, n_cpu=self.n_cpu)

                time1 = perf_counter()

                x_cluster = ex_selector.fit_transform(x_cluster, y_cluster)
                time2 = perf_counter()
                print(
                    "FPGAPredict: Elapsed time for selecting the prediction features is %.4f seconds.\nStart time = %.4f seconds and End time = %.2f seconds\n" % (
                        time2 - time1, time1, time2))
                print(
                    "FPGAPredict: Selecting the prediction features is complete. Min Error = %.2f\n" % ex_selector.get_min_error())
            selected_features_4each_cluster.append(x_cluster.columns.tolist())

        return selected_features_4each_cluster

    '''
        Description:
        Evaluate the overall model on the dataset provided, using the provided estimator and number of clusters (self.n_clusters).

        parameters:
            -x_input: a dataframe of the input features dataset.
            -y_target: a dataframe of the target features dataset.
            -x_clustering: a dataframe of the clustering features dataset.                
            -with_clustering: whether to perform clustering and prediction together (i.e. whether to use ClusteringRegression model)
            -estimator_4each_cluster: 2D list containing the estimator for each cluster. The length of this list must equal self.n_clusters
            -features_4each_cluster: 2D list containing the input features names for each cluster. The length of this list must equal self.n_clusters
            -random_state: integer or None, used for seeding the random generator of KFold. 
                            If random state is an integer, do seeded shuffle for KFold.
                            else, do true shuffle for KFold with random_state=None.

        returns: A dictionary of the results that have the following keys:
                -"results": this key stores a dictionary storing a list of dataframes of the results for each cluster (i.e. real and predicted values, average error, etc.)
                -"stats": this key stores a dictionary storing a list of dataframes of some statistics summary for each cluster (i.e. total average error, stdev..etc.)
                -"avg_training_error": this key stores a list of floats of the total average training errors of n_runs of kfold.
                -"avg_testing_error": this key stores a list of floats of the total average testing error of n_runs kfold.
                -"features_4each_cluster": this key stores the input parameter (features_4each_cluster) - see its definition above.
                -"estimator_4_each_cluster": this key stores the input parameter(estimator_4each_cluster) - see its definition above.
                -"overall_avg_error": this key stores a float, the overall average testing error over the n_runs.
    '''

    def evaluate_model(self, x_input, y_target, x_clustering, with_clustering=False, estimator_4each_cluster=None,
                       features_4each_cluster=None, random_state=None):

        # prepare the estimator and features for each cluster
        if estimator_4each_cluster is None or np.array(estimator_4each_cluster).shape[0] < self.n_clusters:
            estimator_4each_cluster = [self.estimator for i in range(self.n_clusters)]
        if features_4each_cluster is None:
            features_4each_cluster = [self.features for i in range(self.n_clusters)]

        # prepare the results dictioanry that will be returned
        nruns_result_dict = {"results": {}, "stats": {}, "avg_training_error": {}, "avg_testing_error": {},
                             "features_4each_cluster": features_4each_cluster,
                             "clustering_features": x_clustering.columns.tolist(),
                             "estimator_4each_cluster": estimator_4each_cluster,
                             "overall_avg_error": 0}

        if with_clustering or self.n_clusters > 1:

            x_4each_cluster, y_4each_cluster, benchmarks_4each_cluster = self.get_clusters(x_input, y_target,
                                                                                           x_clustering)
        else:
            x_4each_cluster = [x_input]
            y_4each_cluster = [y_target]
            benchmarks_4each_cluster = [x_input.index.tolist()]
        if with_clustering:
            # prepare the estimator and features for each cluster
            models = list(zip(estimator_4each_cluster, features_4each_cluster))
            models = [list(model) for model in models]

            # use ClusteringRegression as the new estimator
            estimator = ClusteringRegression(estimators=models, clustering_features=x_clustering.columns.tolist(),
                                             original_clusters=benchmarks_4each_cluster)

            estimator_4each_cluster = [estimator]

            # treat the whole dataset as one cluster (cluster in the training not on the whoel dataset)
            n_clusters = 1
        else:
            n_clusters = self.n_clusters

        # check whether to do true or seeded shuffle
        if isinstance(random_state, int):
            true_shuffle = False
        else:
            true_shuffle = True
            random_state = None

        # iterate through each cluster
        for c in range(n_clusters):

            # get the estimator, features, the input dataset, and target dataset for each cluster
            if not with_clustering:
                estimator = estimator_4each_cluster[c]
                features = features_4each_cluster[c]
                cluster_x_input = x_4each_cluster[c].copy()[features]
                cluster_y_target = y_4each_cluster[c].copy()
            else:
                cluster_x_input = x_input.copy()
                cluster_y_target = y_target.copy()

            # prepare empty lists for these clusters before starting nruns
            nruns_result_dict["results"][c] = []
            nruns_result_dict["stats"][c] = []
            nruns_result_dict["avg_testing_error"][c] = []
            nruns_result_dict["avg_training_error"][c] = []

            for p in range(self.n_runs):

                # increment random state with every run if we are doing seeded shuffle
                if not true_shuffle:
                    random_state += 1

                # do KFold
                result_dict = self.kfold(cluster_x_input, cluster_y_target, x_clustering, estimator=estimator,
                                         random_state=random_state, with_clustering=with_clustering,
                                         print_bool=(self.print_bool and p == 0))
                # append the results
                nruns_result_dict["avg_testing_error"][c].append(result_dict["testing_error"])
                nruns_result_dict["avg_training_error"][c].append(result_dict["training_error"])
                nruns_result_dict["results"][c].append(result_dict["results"])
                nruns_result_dict["stats"][c].append(result_dict["stats"])
                nruns_result_dict["overall_avg_error"] += result_dict["testing_error"]

            # calculate the overall average error
            nruns_result_dict["overall_avg_error"] = nruns_result_dict["overall_avg_error"] / self.n_runs

        nruns_result_dict["overall_avg_error"] = nruns_result_dict["overall_avg_error"] / n_clusters

        return nruns_result_dict

    '''
        Description:
        Perform KFold one time on the dataset provided and using the provided estimator.

        parameters:
            -x_input: a dataframe of the input features dataset.
            -y_target: a dataframe of the target features dataset.
            -x_clustering: a dataframe of the clustering features dataset.
            -estimator: the estimator to be used to train and predict the target metrics.
            -random_state: integer or None, used for seeding the random generator of KFold. 
                            If random_state=None, do true shuffle for KFold.
                            If random state is an integer, do seeded shuffle for KFold.
            -with_clustering: whether to perform clustering and prediction together (i.e. whether to use ClusteringRegression model)
            -print_bool: whether to print some statistics while doing KFold.

        returns: A dictionary of the results that have the following keys:
                -"results": this key stores a dataframe of the results (i.e. real and predicted values, average error, etc.)
                -"stats": this key stores a dataframe of some statistics summary (i.e. total average error, stdev..etc.)
                -"testing_error": this key stores a float, the total average error of kfold.
    '''

    def kfold(self, x_input, y_target, x_clustering, estimator, random_state=None, with_clustering=False,
              print_bool=True):

        # save features names and benchmarks names
        features_names = np.array(x_input.columns)
        clustering_features_names = np.array(x_clustering.columns)
        benchmarks_names = np.array(x_input.index)
        target_name = np.array(y_target.columns)

        # initialize KFold
        shuffle = True

        # use EqualGroupsKFold with clustering regression (to have equal benchmarks from each cluster in the
        # testing/training)
        if with_clustering:
            kfold = EqualGroupsKFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state,
                                     features=clustering_features_names, n_clusters=self.n_clusters)
        else:
            if len(x_input) < self.n_splits:
                n_splits = len(x_input)
                print("FPGAPredict: The number of splits was modified to become %d" % n_splits)
            else:
                n_splits = self.n_splits
            kfold = KFold(n_splits=n_splits, shuffle=shuffle,
                          random_state=random_state)

        # prepare variables to start KFold
        benchmarks = []  # it has the name of the benchmarks
        results = []  # it has features, fold, real, predicted, error, estimator, equation
        all_errors = []  # it has the error of all benchmarks
        all_testy = []  # it has all tested real target
        all_predy = []  # it has all tested predicted target
        split = []  # it has the current split
        fold = -1  # it has the number of the fold
        training_errors = []  # it has the training error of all benchmarks

        # start KFold
        for train0, test0 in kfold.split(x_input, y_target):
            fold += 1
            train_x, test_x = x_input.iloc[train0], x_input.iloc[test0]
            train_y, test_y = y_target.iloc[train0], y_target.iloc[test0]
            train_x2, test_x2 = x_clustering.iloc[train0], x_clustering.iloc[test0]
            n_testing_benchmarks = len(test_x.index.tolist())
            test_x_before_scaling = np.array(test_x)

            # scale the training and testing data
            if self.scaling != 0 and not with_clustering:
                if self.scaling == 1 and not estimator.__class__.__name__ == "ExponentialRegression":
                    if print_bool:
                        print("FPGAPredict: standard scaling will be performed on the input features\n")
                    scaler = StandardScaler()
                else:
                    if print_bool:
                        print("FPGAPredict: Min Max scaling will be performed on the input features\n")
                    scaler = MinMaxScaler(feature_range=(10, 11))

                train_x = scaler.fit_transform(train_x, train_y)
                test_x = scaler.transform(test_x)

            # append the benchmarks names in the testing
            benchmarks += list(benchmarks_names[test0])

            # append the split (benchmarks names of the split)
            split.append((benchmarks_names[train0], benchmarks_names[train0]))

            # create a copy of the estimator and fit it
            estimator = copy(estimator)
            time1 = perf_counter()

            if with_clustering:
                estimator.fit(train_x, train_y, train_x2)
            else:
                estimator.fit(train_x, train_y)

            time2 = perf_counter()
            elapsed1 = time2 - time1
            if with_clustering:
                train_pred = np.array(estimator.predict(train_x, train_x2))
            else:
                train_pred = np.array(estimator.predict(train_x))
            training_error = np.abs(
                np.divide(train_pred.flatten() - np.array(train_y).flatten(), np.array(train_y).flatten())) * 100
            if print_bool:
                print("FPGAPredict: the training error of this split is %s\n" % (training_error))
            training_errors = training_errors + list(training_error)
            time3 = perf_counter()
            if with_clustering:
                pred_y = estimator.predict(test_x, test_x2)
                features = estimator.features

            else:
                pred_y = estimator.predict(test_x)
            time4 = perf_counter()
            elapsed2 = time4 - time3

            # print timing
            if print_bool:
                print("The training time for fold %d = %.4f Seconds\nTesting time for fold %d = %.4f Seconds\n" % (
                    fold, elapsed1, fold, elapsed2))

            pred_y = np.array(pred_y).flatten()
            test_y = np.array(test_y).flatten()

            all_predy += list(pred_y)
            all_testy += list(test_y)

            errors = np.abs(np.divide((pred_y - test_y), test_y) * 100)
            all_errors += errors.tolist()
            errors = errors.flatten()

            result = list(zip([fold for i in range(errors.shape[0])],
                              test_y, pred_y, errors))
            result_columns = ["fold", "Real " + str(target_name[0]), "Predicted " + str(target_name[0]),
                              "Average Error (%)"]

            if with_clustering:
                clusters = estimator.clusters
                features = estimator.features
                predicted_estimators = estimator.estimators_pred
                for g in range(len(result)):
                    result[g] = list(result[g]) + [clusters[g]]
                    result[g] = list(result[g]) + [predicted_estimators[g]]
                result_columns += ["Cluster", "Estimator"]
            else:
                features = [features_names for o in range(n_testing_benchmarks)]
            # try to get the estimator coefs if it is Linear or Exponential
            try:
                b = estimator.coef_
                a = estimator.intercept_
                if not with_clustering:
                    a = [a for i in range(n_testing_benchmarks)]
                    b = [b for i in range(n_testing_benchmarks)]

                equations = [get_equation(a[i], b[i], features[i]) for i in range(n_testing_benchmarks)]
                for g in range(len(result)):
                    result[g] = list(result[g]) + [equations[g]]

                result_columns += ["Equation"]
            except AttributeError:
                a = None

            if not with_clustering:
                for g in range(len(result)):
                    result[g] = test_x_before_scaling[g].tolist() \
                                + list(result[g])
                result_columns = list(features_names) + result_columns

            results += result

        results_df = pd.DataFrame(results, index=benchmarks, columns=result_columns)
        n = len(all_predy)
        all_predy = np.array(all_predy)
        all_testy = np.array(all_testy)
        RRMSE = np.sqrt((1 / n
                         * np.sum(np.power(np.divide(all_predy - all_testy, all_testy), 2))))

        avg_training_error = np.mean(np.abs(training_errors))
        stats_arr = [[np.mean(all_errors), np.std(all_errors), RRMSE]]
        stats_df = pd.DataFrame(stats_arr, columns=["Average Error (%)", "STDEV Error", "RRMSE (%)"],
                                index=["Summary: "])

        return {"results": results_df, "stats": stats_df, "training_error": avg_training_error,
                "testing_error": np.mean(all_errors)}

    # returns a 2D x and y corresponding to each cluster
    def get_clusters(self, x_input, y_target, x_clustering):
        time1 = perf_counter()

        # initialize kmeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)

        # perform standard scaling on the clustering features
        x_clustering = StandardScaler().fit_transform(x_clustering)

        # perform clustering on the scaled dataset
        kmeans.fit(x_clustering, y_target)

        # get kmeans labels and separate each cluster
        labels = np.array(kmeans.labels_)  # np.array(labels)
        time2 = perf_counter()
        if self.print_bool:
            print("FPGAPredict: Elapsed Time for Clustering is %.4f seconds\n" % (time2 - time1))

        x_4each_cluster = []
        y_4each_cluster = []
        benchmarks_4each_cluster = []
        # get each cluster
        for c in range(self.n_clusters):
            # find the indices of this cluster
            cluster_indices = np.array(np.where(labels == c))[0]

            # get the input features for this cluster
            cluster_x = x_input.iloc[cluster_indices]
            cluster_y = y_target.iloc[cluster_indices]

            x_4each_cluster.append(cluster_x)
            y_4each_cluster.append(cluster_y)
            benchmarks_4each_cluster.append(cluster_x.index.tolist())
        return x_4each_cluster, y_4each_cluster, benchmarks_4each_cluster

    '''
    Description:
        stores the nruns_result_dict from evaluate_model into excel file.
        It stores the average of nruns overall each benchmark.
        Then it stores the estimator, features, and errors of each benchmarks.
        Then it stores the average of average errors, standard deviation, and RRMSE of the nruns.
        Finally it stores up to 10 example runs out of the total nruns. 
    '''

    def store_output(self, nruns_result_dict, sheet_name=None, excelname1=None, excelname2=None, mode='w',
                     with_clustering=False):

        # prepare names for the output file and sheet

        if sheet_name is None:
            sheet_name = "Summary of " + str(self.n_runs) + " Runs"

        if excelname1 is None or (not isinstance(excelname1, str)) or (excelname2 is None) or (
                not isinstance(excelname2, str)):
            board_name = self.excelname.split("_")[0]
            _name = str(self.estimator.__class__.__name__) + "_" + str(board_name) + "_c" + str(self.n_clusters) + "_"

            path = str(self.target_name.tolist()[0]) + "/"

        if excelname1 is None or (not isinstance(excelname1, str)):
            result_name1 = name_file(_name, path, ".xlsx")
        else:
            result_name1 = excelname1

        if excelname2 is None or (not isinstance(excelname2, str)):

            result_name2 = _name + ".xlsx"
        else:
            result_name2 = excelname2

        # initialize excel files
        if mode == 'a':
            writer1 = pd.ExcelWriter(result_name1, mode=mode, if_sheet_exists='overlay', engine="openpyxl")
            writer2 = pd.ExcelWriter(result_name2, mode=mode, if_sheet_exists='overlay', engine="openpyxl")
        else:
            writer1 = pd.ExcelWriter(result_name1, engine="openpyxl")
            writer2 = pd.ExcelWriter(result_name2, engine="openpyxl")

        if with_clustering:
            n_clusters = 1
        else:
            n_clusters = self.n_clusters
        # we need to store up to 10 results from ret.
        # we also need to store the average 50 runs

        # compute the error for each cluster
        error_4each_cluster = []
        error_label_4each_cluster = []
        feature_label_4each_cluster = []
        estimators_label_4each_cluster = []
        summary_4each_cluster = []
        stats_4each_cluster = []
        for c in range(n_clusters):

            results_n_runs = pd.concat(nruns_result_dict["results"][c])
            avg_n_runs = results_n_runs.groupby(level=0).mean()
            if with_clustering:
                avg_n_runs = avg_n_runs[["Real " + str(self.target_name[0]), "Predicted " + str(self.target_name[0]),
                                         "Average Error (%)", "Cluster"]]
            else:
                avg_n_runs = avg_n_runs[["Real " + str(self.target_name[0]), "Predicted " + str(self.target_name[0]),
                                         "Average Error (%)"]]
            std_n_runs = results_n_runs[["Average Error (%)"]].groupby(level=0).std()
            std_n_runs = std_n_runs.rename(columns={"Average Error (%)": "STD of Error"})
            summary_n_runs = pd.concat([avg_n_runs, std_n_runs["STD of Error"]], axis=1)
            stats_n_runs = pd.concat(nruns_result_dict["stats"][c]).groupby(level=0).mean()
            stats_4each_cluster.append(stats_n_runs)
            summary_4each_cluster.append(summary_n_runs)

            if with_clustering:
                for c in range(self.n_clusters):
                    try:
                        summary_avg_c = summary_n_runs[summary_n_runs["Cluster"] == c]
                        error_4each_cluster.append(summary_avg_c["Average Error (%)"].mean())
                    except KeyError:
                        error_4each_cluster.append(np.NaN)
                    error_label_4each_cluster.append("cluster " + str(c) + " Avg Error(%)")
                    feature_label_4each_cluster.append("Cluster {} Features".format(c))
                    estimators_label_4each_cluster.append("Cluster {} Estimator".format(c))
            else:
                error_4each_cluster.append(np.mean(nruns_result_dict["avg_testing_error"][c]))
                error_label_4each_cluster.append("Cluster {} Error (%)".format(c))
                feature_label_4each_cluster.append("Cluster {} Features".format(c))
                estimators_label_4each_cluster.append("Cluster {} Estimator".format(c))
        feature_label_4each_cluster.append("Clustering Features")
        features = [
            nruns_result_dict["features_4each_cluster"][:self.n_clusters] + [nruns_result_dict["clustering_features"]]]
        results_n_runs = pd.concat(summary_4each_cluster)

        # stats dataframe has the overall avg error, average stdev of errors, and overall RMSE
        stats = pd.concat(stats_4each_cluster).groupby(level=0).mean()

        # features dataframe has the features for each cluster
        features = pd.DataFrame(features,
                                columns=feature_label_4each_cluster)

        # estimators dataframe has the estimator for each cluster
        estimators = pd.DataFrame([nruns_result_dict["estimator_4each_cluster"][:self.n_clusters]],
                                  columns=estimators_label_4each_cluster[:self.n_clusters],
                                  index=["Summary: "])

        # errors dataframe has the error for each cluster
        errors = pd.DataFrame([error_4each_cluster], columns=error_label_4each_cluster)

        # generate a plot of the benchmarks against the average error
        plt.figure()
        plt.bar(results_n_runs.index.tolist(), np.array(results_n_runs[["Average Error (%)"]]).flatten())
        plt.xticks(rotation=90)
        plt.xlabel("Benchmarks")
        plt.ylabel("Average {} Runs Error (%)".format(self.n_runs))

        # adjust the plot size
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        plt.tight_layout()

        # save the plot locally
        plt.savefig("summary.png")
        img = openpyxl.drawing.image.Image('summary.png')
        img.anchor = "I1"
        # initialize the row and column indices of the excel file
        row = 0
        column = 0

        # store the results_n_runs first
        results_n_runs = results_n_runs.sort_index()
        results_n_runs.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column)
        results_n_runs.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column)

        # store the plot
        worksheet1 = writer1.sheets[sheet_name]
        worksheet1.add_image(img)
        worksheet2 = writer2.sheets[sheet_name]
        worksheet2.add_image(img)
        row = row + len(results_n_runs.index.tolist()) + 2

        # store the estimators
        estimators.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column)
        estimators.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column)
        column += len(estimators.columns.tolist()) + 2

        # store the features
        features.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column, index=False)
        features.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column, index=False)
        column += len(features.columns.tolist()) + 1

        # store the errors
        errors.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column, index=False)
        errors.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column, index=False)
        column += len(errors.columns.tolist()) + 1

        # store the stats
        stats.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column, index=False)
        stats.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column, index=False)
        column = 0
        row = row + 3

        # now store up to 10 results separately of the n_runs
        for j in range(self.n_runs):
            if j > 10:
                break
            for c in range(n_clusters):
                result = nruns_result_dict["results"][c][j]
                result.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column,
                                index_label="Cluster {}".format(c))
                row = row + len(result.index.tolist()) + 2
                stat = nruns_result_dict["stats"][c][j]
                stat.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column,
                              index_label="Cluster {}".format(c))
                row = row + len(stat.index.tolist()) + 2
            row = row + 1

        writer1.save()
        writer2.save()
        print("FPGAPredict: output file was saved as \"%s\"" % result_name1)

        return result_name1, result_name2, results_n_runs

    '''
    Description:
        This function prints up to 10 results of the n_runs in the all_result_dict parameter.
    '''

    def print_result(self, nruns_result_dict, with_clustering=False):
        if with_clustering:
            n_clusters = 1
        else:
            n_clusters = self.n_clusters
        if self.print_bool:
            for j in range(self.n_runs):
                if j > 10:
                    break
                for c in range(n_clusters):
                    print("cluster {}:\n".format(c))
                    result = nruns_result_dict["results"][c][j]
                    print(result.to_string())
                    stat = nruns_result_dict["stats"][c][j]
                    print(stat.to_string())
                    print("\n")

                print("\n")

    def plot_show(self):
        plt.show()


'''
given a and b vectors along with a features array, this function writes an equation as a string.
'''


def get_equation(_a, _b, _features):
    try:
        a = _a[0]
    except:
        a = 0
    b = _b.tolist()
    features = np.array(_features).tolist()

    equation = "%.2f" % a

    # if isinstance(b[0], list) or type(b[0]) == "ndarray":
    # b = b[0]
    try:
        strr = "%.2f" % b[0]
    except TypeError:
        b = b[0]

    for i in range(len(b)):
        try:
            equation += " + %.2f(%s)" % (b[i], features[i])
        except TypeError:
            equation += " + %.2f(%s)" % (b[0][i], features[i])

    return equation
