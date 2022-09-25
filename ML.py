from feature_selection import *
from datetime import date
from time import perf_counter
from sklearn.metrics import mean_squared_error
from utils import name_file


class FPGAPredict:
    '''
    FPGAPredict: a class that is able to read data from an excelfile,
                select features using PCC and exhaustive search,
                cluster the dataset,
                perform KFold with model validation,
                run n times

    initialization variables:

    excelname: the name of the excel file that contains the data, must end with xls or xlsx.
    estimator: an integer from the following dictionary:
 		    {1: 'LinearRegression', 2: 'ExponentialRegression', 3: 'RandomForestRegressor', 4: 'MLPRegressor', 5: 'DecisionTreeRegressor', 6: 'KNeighborsRegressor'}
    n: the number of top PCC ranked prediction features to pass to permutation, only used when perm=True
	perm: boolean indicates whether to select prediction features by permutation or not
	clustering_n: the number of top PCC ranked CLUSTERING features to pass to permutation, only used when clustering_perm=True
	clustering_perm: boolean indicates whether to select CLUSTERING features by permutation or not
	shuffle: whether to true shuffle the data points with KFold. If False, seeded shuffle is performed instead.
	n_splits: Number of KFold n_splits. If n_splits <= 1 then KFold isn't performed and the results will show the training error only.
	n_clusters: the number of clusters to create. If 1 then clustering is not performed
	random_state: an integer indicating the random state to start with. Only used if shuffle=False.
	printt: whether to print the processing log and the results.
	scaling: an integer, whether to scale the dataset. If 1: use standard scaling, If 2: use MinMax, else don't scale.
	clustering_estimators: an array-like of integers for the estimator of each cluster. Only used when estimator=0 (i.e. clustering regression)
	n_runs: an integer, the number of random runs to perform to record their results. Every run is performed with a different seed to make sure each run leads to a different random split.
	clustering_perm_up_to: an integer or None, the maximum number of clustering features to try in Exhaustive Search. Only used if clustering_perm=True.
	perm_up_to: an integer or None, the maximum number of prediction features to try in Exhaustive Search. Only used if perm=True.
	clustering_perm_min: an integer, the minimum number of clustering features to try in Exhaustive Search. Only used if clustering_perm=True.
	perm_min: an integer, the minimum number of prediction features to try in Exhaustive Search. Only used if perm=True.
	parallel: boolean, whether to perform parallel processing with permutation or not. Only used if either perm=True or clustering_perm=True.
	n_cpu: an integer or None, the number of parallel CPU runs to perform. If None, the code calculates the number of available cores and subtracts 3. Only used if parallel=True and either perm or clustering_perm are True.
	clusters_features: A 2D array that contains the features for each cluster in the following format: [features for c1, features for c2..., features for c<n_clusters>]
    '''

    def __init__(self, params):
        global ESTIMATORS

        self.perm_min = params["perm_min"]  # min features to try in exhaustive search
        self.clustering_perm_min = params["clustering_perm_min"]  # min clustering features to try in exhaustive search
        self.parallel = params["parallel"]  # whether to perform parallel processing with exhaustive search
        self.n_cpu = params["n_cpu"]  # number of cores to use with parallel processes
        self.n_runs = params["n_runs"]  # number of random runs to perform
        self.perm_up_to = params["perm_max"]  # max features to try in exhaustive search
        self.clustering_perm_up_to = params[
            "clustering_perm_max"]  # max clustering features to try in exhaustive search
        estimator = params["estimator"]
        if isinstance(estimator, int):  # get the estimator from the dictionary
            self.estimator = ESTIMATORS[estimator]
        else:
            self.estimator = estimator

        self.printt = params["print"]  # whether to print the results or not
        self.excelname = params["excelname"]  # the name of the excel file that has the data
        self.random_state = params["random_state"]  # random state with KFold
        self.perm = params["perm"]  # boolean, whether to do prediction exhaustive search or not
        self.n = params["n"]  # No. features to pass to exhaustive search. only used if perm is True
        self.clustering_n = params[
            "clustering_n"]  # No. clustering features to pass to exhaustive search. only used if clustering_perm is True
        self.clustering_perm = params["clustering_perm"]  # boolean, whether to do clustering exhaustive search or not
        self.shuffle = params["shuffle"]  # whether to perform true shuffle or not. If not, seeded shuffle is performed
        self.board = params["excelname"].split("_")[0]  # the board name extracted from the excel file
        self.benchmarks = None  # name of the benchmarks
        self.n_clusters = params["n_clusters"]  # number of clusters to create
        self.n_splits = params["n_splits"]  # number of splits with KFold
        self.clustering_features = None  # features used to cluster the dataset
        self.clustering_estimators = params["clustering_estimators"]  # estimators used for each cluster
        self.clusters_features = params["clusters_features"]  # features used for each cluster
        self.top_clustering_features = []
        # if the estimator is clustering regression, set up the features and estimators for each cluster
        if params["estimator"] == 0:
            clustering_estimators = zip(list(self.clustering_estimators), list(self.clusters_features))
            self.clustering_estimators = list(clustering_estimators)
            self.estimator.set_parameters(estimators=self.clustering_estimators)
        self.scaling = params["scaling"]
        self.results = None # a dataframe of the results, created after a call to store_output

    # start FPGAPredict based on the configurations
    def run(self, features=None, target=None, clustering_features=None, output_results=True):
        if self.printt:
            print("******** FPGAPredict Started Running ******** \n")

        # assign the clustering features for clustering regression
        if self.estimator.__class__.__name__ == "ClusteringRegression":
            self.estimator.set_parameters(clustering_features=clustering_features)

        # load data as data frames
        x, y, x2 = self.read_excel_sheet(features, target, clustering_features)

        # initialize results dictionary
        all_args = {"features": [], "results": [], "split": [], "errors": [], "avg_training_error": []}

        # save clustering features
        if clustering_features is not None:
            self.clustering_features = list(clustering_features)

        # find clustering features using permutation.
        if self.clustering_perm and self.n_clusters > 1:
            print("selecting clustering features by passing the top %d features to permutation.." % self.clustering_n)

            # initialize Exhaustive Search using KMeans
            perm = ExhaustiveSearch(estimator=KMeans(n_clusters=self.n_clusters, random_state=42),
                                    n_splits=self.n_splits, n=self.clustering_n, clustering=True,
                                    max=self.clustering_perm_up_to, min=self.clustering_perm_min,
                                    parallel=self.parallel, n_cpu=self.n_cpu, printt=self.printt)

            time1 = perf_counter()

            if self.printt:
                print("Starting exhaustive search. Current time is %.4f seconds" % time1)

            # select clustering features by calling fit_transform()
            x2 = perm.fit_transform(x2, y, x)  # transforms clustering features to the top choice
            time2 = perf_counter()
            self.clustering_features = x2.columns.tolist()
            if self.printt:
                print("Finished exhaustive search. Current time is %.2f seconds" % time2)
            elapsed = time2 - time1
            self.top_clustering_features = np.array(perm.best_c_subsets["Clustering Features"])

            print(
                "ELAPSED TIME FOR SELECTING FEATURES IS: %.4f seconds.\nStart Time = %.4f seconds and End Time = %.4f seconds\n" % (
                    elapsed, time1, time2))
            print("clustering features were selected successfully")
            print("clustering features selected are: \n%s" % x2.columns.tolist())

        if self.n_clusters > 1 and self.estimator.__class__.__name__ != "ClusteringRegression":
            print("-> Processing with clustering features: %s" % (self.clustering_features))
            time1 = perf_counter()

            # initialize kmeans
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)

            # perform standard scaling on the clustering features
            x2 = StandardScaler().fit_transform(x2)

            # perform clustering on the scaled dataset
            kmeans.fit(x2, y)

            # get kmeans labels and separate each cluster
            labels = np.array(kmeans.labels_)  # np.array(labels)
            time2 = perf_counter()
            if self.printt:
                print("Elapsed Time for Clustering is %.4f seconds\n" % (time2 - time1))

            # start processing each cluster separately
            for c in range(self.n_clusters):
                if self.printt:
                    print("\nProcessing cluster %d...\n" % c)

                # find the indices of this cluster
                cluster_indices = np.array(np.where(labels == c))[0]

                # get the input features for this cluster
                cluster_x = pd.DataFrame(x.copy().to_numpy()[cluster_indices], columns=x.columns,
                                         index=x.index[cluster_indices])

                # if clusters_features are declared then get the features for this cluster only
                if self.clusters_features is not None:
                    cluster_x = cluster_x[self.clusters_features[c]]

                # get the output feature for this cluster
                cluster_y = pd.DataFrame(y.copy().to_numpy()[cluster_indices], columns=y.columns,
                                         index=y.index[cluster_indices])

                # if n_splits is 1 or less, fit and predict the whole dataset without KFold
                if self.n_splits <= 1:
                    self.select_predict_alldataset(cluster_x, cluster_y, f)
                    all_args = self.args
                    all_args["results"] = [all_args["results"]]
                    all_args["errors"] = [all_args["errors"]]
                    all_args["features"] = [all_args["features"]]
                    return all_args

                else:  # perform KFold
                    ret = self.select_predict(cluster_x, cluster_y)

                    all_args["results"].append(
                        self.args["results"])
                    all_args["errors"].append(self.args["errors"])
                    all_args["features"].append(self.args["features"])

            return all_args

        # perform KFold without clustering
        elif self.n_splits > 1:
            if self.clusters_features is not None and self.estimator.__class__.__name__ != "ClusteringRegression":
                x = x[self.clusters_features[0]]
            ret = self.select_predict(x, y, x2)
            all_args = self.args
            all_args["results"] = [all_args["results"]]
            all_args["errors"] = [all_args["errors"]]
            all_args["features"] = [all_args["features"]]
            return all_args
        else:  # whole dataset without clustering
            self.select_predict_alldataset(x, y)
            all_args = self.args
            all_args["results"] = [all_args["results"]]
            all_args["errors"] = [all_args["errors"]]
            all_args["features"] = [all_args["features"]]
            return all_args

    # reads the excel file and returns x and y as numpy arrays
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

        # save the target name
        self.target_feature = target

        # run with xlrd if the file is xls, otherwise run with openpyxl
        if self.excelname.split(".")[1] == "xls":
            engine = "xlrd"
        else:
            engine = "openpyxl"

        features_df = pd.read_excel(self.excelname, sheet_name="features", engine=engine, index_col=0,
                                    usecols=features)  # read features sheet
        target_df = pd.read_excel(self.excelname, sheet_name="target", engine=engine, index_col=0,
                                  usecols=target)  # read target sheet
        clustering_features = pd.read_excel(self.excelname,
                                            sheet_name="clustering_features", engine=engine, index_col=0,
                                            usecols=clustering)  # read clustering features

        self.clustering_features = clustering_features.columns.tolist()
        x = features_df
        y = target_df
        x2 = clustering_features

        # save features and benchmarks names
        self.features = features_df.columns
        self.benchmarks = x.index.tolist()
        self.target_name = target_df.columns
        return x, y, x2

    def select_predict_alldataset(self, x, y):
        print("FPGAPredict: The results will be computed on the whole dataset")
        if self.perm:
            perm = ExhaustiveSearch(estimator=self.estimator, shuffle=self.shuffle, n_splits=self.n_splits,
                                    n=self.n, random_state=self.random_state, parallel=self.parallel, n_cpu=self.n_cpu, printt=self.printt)
            x = perm.fit_transform(x, y)
            
            print("Selecting Features is complete.\n")
            print("Selected features are: " % x.columns)

        self.features = x.columns

        numpy_x = x.copy().to_numpy()
        numpy_y = y.copy().to_numpy().flatten()

        benchmarks = x.index.tolist()
        self.estimator = copy(self.estimator)  # create a copy in memory
        self.estimator.fit(numpy_x, numpy_y)
        pred_y = np.array(self.estimator.predict(numpy_x)).flatten()
        all_errors = np.divide(np.abs(pred_y - numpy_y), numpy_y) * 100
        if self.estimator.__class__.__name__ == "ClusteringRegression":
            b = self.estimator.coef_
            a = self.estimator.intercept_

            features = self.estimator.features
            clusters = self.estimator.clusters
            estimators = self.estimator.estimators_pred
            if a is not None and b is not None:
                equation = get_equation(a, b, features)
                equations = [equation for i in range(all_errors.shape[0])]
                results = list(
                    zip(numpy_y, pred_y, all_errors,
                        estimators, clusters,
                        equations))
            else:
                equation = None
                results = list(
                    zip(numpy_y, pred_y, all_errors,
                        estimators, clusters))

        elif self.estimator.__class__.__name__ == "LinearRegression" or self.estimator.__class__.__name__ == "ExponentialRegression":
            clusters = None
            b = self.estimator.coef_
            a = self.estimator.intercept_
            if b is not None:
                equation = get_equation(a, b, self.features)
                results = list(
                    zip(numpy_y, pred_y, all_errors,
                        [self.estimator.__class__.__name__ for i in range(len(benchmarks))],
                        [equation for i in range(len(benchmarks))]))
            else:
                clusters = None
                equation = None
                results = list(
                    zip(numpy_y, pred_y, all_errors,
                        [self.estimator.__class__.__name__ for i in range(len(benchmarks))]))
        else:
            clusters = None
            equation = None
            results = list(
                zip(numpy_y, pred_y, all_errors,
                    [self.estimator.__class__.__name__ for i in range(len(benchmarks))]))

        columns = []
        for target in self.target_name:
            columns += ["Real " + str(target), "Predicted " + str(target)]
            self.target = target
        columns += ["Error (%)", "Estimator"]
        if clusters is not None:
            columns += ["Cluster"]
        if equation is not None:
            columns += ["Equation"]
        results_df = pd.DataFrame(results, index=benchmarks, columns=columns)

        errors_arr = [[np.mean(all_errors), np.std(all_errors)]]
        errors_df = pd.DataFrame(errors_arr, columns=["Average Error (%)", "STDEV Error"],
                                 index=["Summary: "])
        self.args = {"features": self.features, "results": [results_df], "errors": [errors_df], "split": None,
                     "avg_training_error": np.mean(all_errors)}

    # select features (if enabled by self.perm) and perform cross validation
    def select_predict(self, _x, y, x2=None, features=None):
        print("running with estimator: " + str(self.estimator))

        if x2 is None:  # clustering features
            x2 = _x
        else:
            c_features = x2.columns.tolist()

        # transform input features if features is not None
        if features is not None:
            x = _x[features]
        else:
            x = _x.copy()

        # perform Exhaustive Search for prediction
        if self.perm:
            perm = ExhaustiveSearch(estimator=self.estimator, n_splits=self.n_splits,
                                    n=self.n, scaling=self.scaling, max=self.perm_up_to,
                                    parallel=self.parallel, n_cpu=self.n_cpu, min=self.perm_min)
            time1 = perf_counter()

            x = perm.fit_transform(x, y)
            time2 = perf_counter()
            print(
                "Elapsed time for selecting features is %.4f seconds.\nStart time = %.4f seconds and End time = %.2f seconds\n" % (
                    time2 - time1, time1, time2))
            print("Selecting Features is complete. Min Error = %.2f\n" % perm.get_min_error())

        self.features = x.columns

        numpy_x = x.copy().to_numpy()
        numpy_y = y.copy().to_numpy()
        numpy_x2 = x2.copy().to_numpy()

        # initialize variables to start KFold with n runs
        summaries = []
        all_errors_df = []
        avg_total_training_error = 0
        avg_total_testing_error = 0

        # always shuffle the dataset (seeded or true shuffle)
        shuffle = True

        # check whether to do true shuffle or seeded shuffle
        if self.shuffle:
            random_state = None
        else:
            random_state = self.random_state

        # find the original clusters if the estimator is ClusteringRegression
        if self.estimator.__class__.__name__ == "ClusteringRegression":
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            # find the original clusters
            c_x = StandardScaler().fit_transform(x2)
            kmeans.fit(c_x)
            labels = kmeans.labels_
            original_clusters = {}  # dictionary of the clusters
            benchmarks = x2.index.tolist()

            for i in range(len(list(labels))):
                try:
                    original_clusters[labels[i]].append(benchmarks[i])
                except KeyError:
                    original_clusters[labels[i]] = [benchmarks[i]]
            self.estimator.set_parameters(original_clusters=original_clusters)

        # start 50 runs
        for p in range(self.n_runs):

            if not self.shuffle:  # increment random state if seeded shuffle
                random_state += 1

            # use EqualGroupsKFold with clustering regression (to have equal benchmarks from each cluster in the testing/training)
            if self.estimator.__class__.__name__ == "ClusteringRegression":
                kfold = EqualGroupsKFold(n_splits=self.n_splits, shuffle=shuffle, random_state=random_state,
                                         features=self.clustering_features, n_clusters=self.n_clusters)
            else:
                if len(x) < self.n_splits:
                    n_splits = len(x)
                    print("Number of splits was modified to become %d" % n_splits)
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
            for train0, test0 in kfold.split(x, y):
                fold += 1
                train_x, test_x = numpy_x[train0], numpy_x[test0]
                train_y, test_y = numpy_y[train0], numpy_y[test0]
                train_x2, test_x2 = numpy_x2[train0], numpy_x2[test0]
                test_x_before_scaling = np.copy(test_x)

                # scale the training and testing data
                if self.scaling != 0 and self.estimator.__class__.__name__ != "ClusteringRegression":
                    if self.scaling == 1 and self.estimator.__class__.__name__ != "ExponentialRegression":
                        if self.printt and p < 10:
                            print("FPGAPredict: standard scaling will be performed on the input features\n")
                        scaler = StandardScaler()
                    else:
                        if self.printt and p < 10:
                            print("FPGAPredict: Min Max scaling will be performed on the input features\n")
                        scaler = MinMaxScaler(feature_range=(10, 11))

                    train_x = scaler.fit_transform(train_x, train_y)
                    test_x = scaler.transform(test_x)

                # convert back to pandas dataframes
                train_x = pd.DataFrame(train_x, columns=self.features, index=x.index[train0])
                train_y = pd.DataFrame(train_y, columns=self.target_name, index=x.index[train0])
                test_x = pd.DataFrame(test_x, columns=self.features, index=x.index[test0])

                # append the benchmarks names in the testing
                benchmarks += x.index[test0].tolist()

                # append the split (benchmarks names of the split)
                split.append((x.index[train0].tolist(), x.index[test0].tolist()))

                # create a copy of the estimator and fit it
                self.estimator = copy(self.estimator)
                time1 = perf_counter()
                if self.estimator.__class__.__name__ == "ClusteringRegression":
                    train_x2 = pd.DataFrame(train_x2, columns=c_features, index=x.index[train0])
                    self.estimator.fit(train_x, train_y, train_x2)
                else:
                    self.estimator.fit(train_x, train_y)
                time2 = perf_counter()
                elapsed1 = time2 - time1
                if self.estimator.__class__.__name__ == "ClusteringRegression":
                    train_pred = np.array(self.estimator.predict(train_x, train_x2))
                else:
                    train_pred = np.array(self.estimator.predict(train_x))
                training_error = np.abs(
                    np.divide(train_pred.flatten() - np.array(train_y).flatten(), np.array(train_y).flatten())) * 100
                if self.printt and p < 10:
                    print("training error of this split is %s" % (training_error))
                training_errors = training_errors + list(training_error)
                time3 = perf_counter()
                if self.estimator.__class__.__name__ == "ClusteringRegression":
                    test_x2 = pd.DataFrame(test_x2, columns=c_features, index=x.index[test0])
                    pred_y = self.estimator.predict(test_x, test_x2)
                else:
                    pred_y = self.estimator.predict(test_x)
                time4 = perf_counter()
                elapsed2 = time4 - time3

                # print timing
                if self.printt and p < 10:
                    print("Training time for fold %d = %.4f Seconds\nTesting time for fold %d = %.4f Seconds\n" % (
                        fold, elapsed1, fold, elapsed2))
                all_predy += list(pred_y.flatten())
                all_testy += list(test_y.flatten())
                pred_y = pred_y.flatten()
                test_y = test_y.flatten()
                errors = np.abs(np.divide((pred_y - test_y), test_y) * 100)
                all_errors += errors.tolist()

                test_y = test_y.flatten()
                pred_y = pred_y.flatten()
                errors = errors.flatten()

                # prepare a results list with the input features, target features, fold, error, cluster, estimator..etc
                if self.estimator.__class__.__name__ == "ClusteringRegression":
                    b = self.estimator.coef_
                    a = self.estimator.intercept_
                    equation = 1

                    features = self.estimator.features
                    clusters = self.estimator.clusters
                    estimators = self.estimator.estimators_pred

                    if a is not None and b is not None:
                        equations = [get_equation(a[i], b[i], features[i]) for i in range(errors.shape[0])]

                        result = list(
                            zip([fold for i in range(errors.shape[0])],
                                test_y, pred_y, errors,
                                estimators, clusters,
                                equations))
                    else:
                        equation = None
                        result = list(
                            zip([fold for i in range(errors.shape[0])],
                                test_y, pred_y, errors,
                                estimators, clusters))

                elif self.estimator.__class__.__name__ == "LinearRegression" or self.estimator.__class__.__name__ == "ExponentialRegression":
                    clusters = None
                    b = self.estimator.coef_
                    a = self.estimator.intercept_
                    if b is not None:
                        equation = get_equation(a, b, self.features)
                        result = list(
                            zip([fold for i in range(errors.shape[0])], test_y, pred_y, errors,
                                [self.estimator.__class__.__name__ for i in range(errors.shape[0])],
                                [equation for i in range(errors.shape[0])]))
                    else:
                        clusters = None
                        equation = None
                        result = list(
                            zip([fold for i in range(errors.shape[0])], test_y, pred_y, errors,
                                [self.estimator.__class__.__name__ for i in range(errors.shape[0])]))
                else:
                    clusters = None
                    equation = None
                    result = list(
                        zip([fold for i in range(errors.shape[0])], test_y, pred_y, errors,
                            [self.estimator.__class__.__name__ for i in range(errors.shape[0])]))
                if not self.estimator.__class__.__name__ == "ClusteringRegression":
                    for g in range(len(result)):
                        result[g] = test_x_before_scaling[g].tolist() + list(result[g])

                results += result

            if not self.estimator.__class__.__name__ == "ClusteringRegression":
                columns = list(self.features) + ["fold"]
            else:
                columns = ["fold"]
            for target in self.target_name:
                columns += ["Real " + str(target), "Predicted " + str(target)]
                self.target = target
            columns += ["Average Error (%)", "Estimator"]
            if clusters is not None:
                columns += ["Cluster"]
            if equation is not None:
                columns += ["Equation"]

            results_df = pd.DataFrame(results, index=benchmarks, columns=columns)
            n = len(all_predy)
            all_predy = np.array(all_predy)
            all_testy = np.array(all_testy)
            RRMSE = np.sqrt((1 / n
                             * np.sum(np.power(np.divide(all_predy - all_testy, all_testy), 2))))

            avg_training_error = np.mean(np.abs(training_errors))
            avg_total_training_error += avg_training_error
            avg_total_testing_error += np.mean(all_errors)
            errors_arr = [[np.mean(all_errors), np.std(all_errors), RRMSE]]
            errors_df = pd.DataFrame(errors_arr, columns=["Average Error (%)", "STDEV Error", "RRMSE (%)"],
                                     index=["Summary: "])

            all_errors_df.append(errors_df)  # append all error rates for each run
            summaries.append(results_df)  # append results for each run

            if self.printt and p < 10:
                print()
                print("----------- Result for the run No. %d of this cluster -----------\n" % (p))
                print("Passed features -> " + str(list(self.features)))
                print()
                print(results_df.to_string())
                print()
                print(errors_df.to_string())
                print("-------------------------------------------")
                print()
        avg_total_training_error = avg_total_training_error / self.n_runs
        avg_total_testing_error = avg_total_testing_error / self.n_runs
        self.error = avg_total_testing_error
        self.args = {"features": self.features, "results": summaries, "split": None, "errors": all_errors_df,
                     "avg_training_error": avg_total_training_error}

        return self.args

    '''
    gets the average error of n runs. Use this function only after a previous call to fit() or select_predict()
    '''

    def get_avg_error(self):
        return self.error

    '''
    store the output using an argument array (result from fit())
    all_args: a dictionary that contains two keys: results and errors. Each key contains a list for the output of each cluster, each output contains results (as an array) for n runs.
    n_clusters: the number of clusters.
    excelname1: the name of the first excel file.
    excelname2: the name of the second excel file
    sheet_name: name of the sheet.
    mode: whether to write or append. 
    '''

    def store_output(self, all_args, n_clusters=None, excelname1=None, excelname2=None, sheet_name=None, mode='w'):

        if n_clusters is None:
            n_clusters = self.n_clusters
        if sheet_name is None:
            sheet_name = "Summary of " + str(self.n_runs) + " Runs"

        if excelname1 is None or (not isinstance(excelname1, str)) or (excelname2 is None) or (
                not isinstance(excelname2, str)):
            # dd/mm/YY
            today = date.today()

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

        if mode == 'a':
            writer1 = pd.ExcelWriter(result_name1, mode=mode, if_sheet_exists='overlay', engine="openpyxl")
            writer2 = pd.ExcelWriter(result_name2, mode=mode, if_sheet_exists='overlay', engine="openpyxl")
        else:
            writer1 = pd.ExcelWriter(result_name1, engine="openpyxl")
            writer2 = pd.ExcelWriter(result_name2, engine="openpyxl")

        row = 0
        errors = []  # average error of each cluster
        features = []
        if self.estimator.__class__.__name__ == "ClusteringRegression" or self.clusters_features is not None:
            features = self.clusters_features
        labels = []
        labels2 = ["cluster " + str(i) + " Features" for i in range(n_clusters)]
        if n_clusters > 1:
            labels2.append("clustering features")
        all_summaries = []

        if self.n_runs < 10:
            J = self.n_runs
        else:
            J = 10
        for cluster in range(len(all_args["results"])):
            summaries_avg = pd.concat(all_args["results"][cluster]).groupby(level=0).mean()
            if self.estimator.__class__.__name__ == "ClusteringRegression":
                summaries_avg = summaries_avg[
                    ["Real " + str(self.target), "Predicted " + str(self.target), "Average Error (%)", "Cluster"]]
                errors = []
                labels = []
                for c in range(n_clusters):
                    summaries_avg_c = summaries_avg[summaries_avg["Cluster"] == c]
                    errors.append(summaries_avg_c["Average Error (%)"].mean())
                    labels.append("cluster " + str(c) + " Avg Error(%)")
            else:
                summaries_avg = summaries_avg[
                    ["Real " + str(self.target), "Predicted " + str(self.target), "Average Error (%)"]]

            summaries_std = pd.concat(all_args["results"][cluster]).groupby(level=0).std()

            summaries_std = summaries_std.rename(columns={"Average Error (%)": "STD of Error"})

            summaries_avg = pd.concat([summaries_avg, summaries_std["STD of Error"]], axis=1)
            all_summaries.append(summaries_avg)
            avg_error = summaries_avg["Average Error (%)"].mean()
            if self.estimator.__class__.__name__ != "ClusteringRegression":
                errors.append(avg_error)
                labels.append("cluster " + str(cluster) + " Avg Error(%)")
            if self.estimator.__class__.__name__ != "ClusteringRegression" and self.clusters_features is None:
                features.append(list(all_args["features"][cluster]))

        clusters_features = list(features)
        if n_clusters > 1 and self.estimator.__class__.__name__ != "ClusteringRegression":
            features.append(self.clustering_features)

        estimator_name = self.estimator.__class__.__name__
        if estimator_name == "ClusteringRegression":
            estimator_name = self.estimator.estimators[0][0].__class__.__name__

        features = pd.DataFrame([features], columns=labels2, index=[estimator_name])
        errors = pd.DataFrame([errors], columns=labels, index=[""])
        results = pd.concat(all_summaries)
        n = len(results.index.tolist())
        all_predy = results["Predicted " + str(self.target)].to_numpy()
        all_testy = results["Real " + str(self.target)].to_numpy()

        RRMSE = np.sqrt((1 / n
                         * np.sum(np.power(np.divide(all_predy - all_testy, all_testy), 2))))

        avg_error = results["Average Error (%)"].mean()
        std_error = results["Average Error (%)"].std()
        stats = [[avg_error, std_error, RRMSE]]
        stats = pd.DataFrame(stats, columns=["Average " + str(self.n_runs) + " Runs Error (%)", "STD of Avg Errors",
                                             "RRMSE"], index=["Summary:"])
        results = results.sort_index()
        self.results = results
        results.to_excel(writer1, startrow=row, sheet_name=sheet_name,
                         index_label="Summary of {} runs".format(self.n_runs))
        results.to_excel(writer2, startrow=row, sheet_name=sheet_name,
                         index_label="Summary of {} runs".format(self.n_runs))
        row += n + 2
        column = 0
        features.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column)
        features.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column)
        column += len(features.columns.tolist()) + 1

        errors.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column)
        errors.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column)
        column += len(errors.columns.tolist()) + 1
        stats.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column)
        stats.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column)
        row = row + 4
        for run in range(J):

            for cluster in range(len(all_args["results"])):
                all_args["results"][cluster][run].to_excel(writer1, startrow=row, sheet_name=sheet_name,
                                                           index_label="cluster " + str(cluster))
                all_args["results"][cluster][run].to_excel(writer2, startrow=row, sheet_name=sheet_name,
                                                           index_label="cluster " + str(cluster))
                row += len(all_args["results"][cluster][run].index.tolist()) + 2
                all_args["errors"][cluster][run].to_excel(writer1, startrow=row, sheet_name=sheet_name,
                                                          index_label="cluster " + str(cluster))
                all_args["errors"][cluster][run].to_excel(writer2, startrow=row, sheet_name=sheet_name,
                                                          index_label="cluster " + str(cluster))
                row += len(all_args["errors"][cluster][run].index.tolist()) + 2
            row += 2

        print("*********** Summary of %d runs ***********\n" % self.n_runs)
        print(stats.to_string())
        print()
        print("Estimator and features used:\n")
        print(features.to_string())
        print()
        print("Average Error(s) for Each Cluster:")
        print(errors.to_string())
        print()

        writer1.save()
        writer2.save()

        print("FPGAPredict: Results were saved in more detail as %s" % result_name1)

        return result_name1, result_name2, clusters_features


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
