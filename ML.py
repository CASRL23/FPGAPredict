import numpy as np

from feature_selection import *
from datetime import date
from time import perf_counter
from utils import name_file

class FPGAPredict:

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

        self.print_bool = params["print"]  # whether to print the results or not
        self.excelname = params["excelname"]  # the name of the excel file that has the data
        self.perm = params["perm"]  # boolean, whether to do prediction exhaustive search or not
        self.n = params["n"]  # No. features to pass to exhaustive search. only used if perm is True
        self.clustering_n = params[
            "clustering_n"]  # No. clustering features to pass to exhaustive search. only used if clustering_perm is True
        self.clustering_perm = params["clustering_perm"]  # boolean, whether to do clustering exhaustive search or not
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
        self.results = None  # a dataframe of the results, created after a call to store_output

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

        # run with xlrd if the file is xls, otherwise run with openpyxl
        if self.excelname.split(".")[1] == "xls":
            engine = "xlrd"
        else:
            engine = "openpyxl"

        x_input = pd.read_excel(self.excelname, sheet_name="features", engine=engine, index_col=0,
                                    usecols=features)  # read features sheet
        y_target = pd.read_excel(self.excelname, sheet_name="target", engine=engine, index_col=0,
                                  usecols=target)  # read target sheet
        x_clustering = pd.read_excel(self.excelname,
                                            sheet_name="features", engine=engine, index_col=0,
                                            usecols=clustering)  # read clustering features

        self.clustering_features = x_clustering.columns.tolist()

        # save features and benchmarks names
        self.features = x_input.columns.tolist()
        self.benchmarks = x_input.index.tolist()
        self.target_name = y_target.columns
        return x_input, y_target, x_clustering

    def select_clustering_features(self, x, y, x2):
        if self.clustering_perm:
            print(
                "FPGAPredict: selecting clustering features by passing the top %d features to exhaustive search.." % self.clustering_n)

            # initialize Exhaustive Search using KMeans
            perm = ExhaustiveSearch(estimator=KMeans(n_clusters=self.n_clusters, random_state=42),
                                    n_splits=self.n_splits, n=self.clustering_n, clustering=True,
                                    max=self.clustering_perm_up_to, min=self.clustering_perm_min,
                                    parallel=self.parallel, n_cpu=self.n_cpu, printt=self.print_bool)

            time1 = perf_counter()

            if self.print_bool:
                print("FPGAPredict: Starting exhaustive search. Current time is %.4f seconds" % time1)

            # start exhaustive search to select clustering features by calling fit()
            perm.fit(x2, y, x)  # transforms clustering features to the top choice
            time2 = perf_counter()
            if self.print_bool:
                print("FPGAPredict: Finished exhaustive search. Current time is %.2f seconds" % time2)
            elapsed = time2 - time1
            top_clustering_features = np.array(perm.best_c_subsets["Clustering Features"])

            print(
                "FPGAPredict: ELAPSED TIME FOR SELECTING THE FEATURES IS: %.4f seconds.\nStart Time = %.4f seconds and End Time = %.4f seconds\n" % (
                    elapsed, time1, time2))
            print("FPGAPredict: The clustering features were selected successfully")
            print("FPGAPredict: The selected clustering features are: \n%s" % top_clustering_features[0])
        else:
            top_clustering_features = [x2.columns.tolist()]
        return top_clustering_features

    def select_prediction_features(self, _x_input, _y_target, _x2_clustering, pred_features=None, clustering_features=None):
        print("FPGAPredict: Selecting the prediction features for each cluster ..")
        x_input = _x_input.copy()
        y_target = _y_target.copy()
        x_clustering = _x2_clustering.copy()
        if pred_features is not None:
            if self.print_bool:
                print("FPGAPredict: Processing with the prediction features %s" %pred_features)
            x_input = x_input[pred_features]
        if clustering_features is not None:
            if self.print_bool:
                print("FPGAPredict: Processing with the clustering features %s" %clustering_features)
            x_clustering = x_clustering[clustering_features]

        x_4each_cluster, y_4each_cluster, benchmarks_4each_cluster = self.get_clusters(x_input,y_target,x_clustering)

        # prepare an array of selected features
        selected_features_4each_cluster = []
        cluster = -1
        for x_cluster, y_cluster in list(zip(x_4each_cluster, y_4each_cluster)):
            cluster += 1

            if self.print_bool:
                print("\nFPGAPredict: Processing cluster %d ...\n" %cluster)
            # perform Exhaustive Search for prediction
            if self.perm:
                perm = ExhaustiveSearch(estimator=self.estimator, n_splits=self.n_splits,
                                        n=self.n, scaling=self.scaling, max=self.perm_up_to,
                                        parallel=self.parallel, n_cpu=self.n_cpu,
                                        min=self.perm_min, printt=self.print_bool)
                time1 = perf_counter()

                x_cluster = perm.fit_transform(x_cluster, y_cluster)
                time2 = perf_counter()
                print(
                    "FPGAPredict: Elapsed time for selecting the prediction features is %.4f seconds.\nStart time = %.4f seconds and End time = %.2f seconds\n" % (
                        time2 - time1, time1, time2))
                print(
                    "FPGAPredict: Selecting the prediction features is complete. Min Error = %.2f\n" % perm.get_min_error())
            selected_features_4each_cluster.append(x_cluster.columns.tolist())

        return selected_features_4each_cluster

    # performs KFold with n runs
    def validate_model(self, x_input, y_target, x_clustering, with_clustering=False, estimators_4each_cluster=None, features_4each_cluster=None, true_shuffle=True, random_state=None):
        if estimators_4each_cluster is None:
            estimators_4each_cluster = [self.estimator for i in range(self.n_clusters)]
        if features_4each_cluster is None:
            features_4each_cluster = [self.features for i in range(self.n_clusters)]

        ret = {"results": {}, "stats": {}, "avg_training_error": {}, "avg_testing_error": {},
               "features_4each_cluster": features_4each_cluster, "estimator_4each_cluster": estimators_4each_cluster,
               "overall_avg_error": 0}

        self.true_shuffle = true_shuffle
        self.random_state = random_state

        models = list(zip(estimators_4each_cluster, features_4each_cluster))
        models = [list(model) for model in models]
        if with_clustering or self.n_clusters > 1:

            x_4each_cluster, y_4each_cluster, benchmarks_4each_cluster = self.get_clusters(x_input, y_target, x_clustering)
        else:
            x_4each_cluster = [x_input]
            y_4each_cluster = [y_target]
            benchmarks_4each_cluster = [x_input.index.tolist()]
        if with_clustering:
            estimator = ClusteringRegression(estimators=models, clustering_features=x_clustering.columns.tolist(),
                                             original_clusters=benchmarks_4each_cluster)
            estimators_4each_cluster = [estimator]
            n_clusters = 1
        else:
            n_clusters = self.n_clusters

        if self.true_shuffle:
            random_state = None
        else:
            if isinstance(self.random_state, int):
                random_state = self.random_state
            else:
                random_state = 0

        # iterate through each cluster
        for c in range(n_clusters):

            estimator = estimators_4each_cluster[c]
            features = features_4each_cluster[c]
            cluster_x_input = x_4each_cluster[c].copy()[features]
            cluster_y_target = y_4each_cluster[c].copy()
            ret["results"][c] = []
            ret["stats"][c] = []
            ret["avg_testing_error"][c] = []
            ret["avg_training_error"][c] = []

            for p in range(self.n_runs):
                if not self.true_shuffle:
                    random_state += 1

                result_dict = self.validate_kfold(cluster_x_input, cluster_y_target, x_clustering, estimator=estimator, random_state=random_state, with_clustering=with_clustering, print_bool=(self.print_bool and p == 0))
                ret["avg_testing_error"][c].append(result_dict["testing_error"])
                ret["avg_training_error"][c].append(result_dict["training_error"])
                ret["results"][c].append(result_dict["results"])
                ret["stats"][c].append(result_dict["stats"])
                ret["overall_avg_error"] += result_dict["testing_error"]
            ret["overall_avg_error"] = ret["overall_avg_error"]/self.n_runs
        ret["overall_avg_error"] = ret["overall_avg_error"]/n_clusters
        return ret

    '''
    
    '''
    def validate_kfold(self, x_input, y_target, x_clustering, estimator, random_state=None, with_clustering=False, print_bool=True):

        # save features names and benchmarks names
        features_names = np.array(x_input.columns)
        clustering_features_names = np.array(x_clustering.columns)
        benchmarks_names = np.array(x_input.index)
        target_name = np.array(y_target.columns)

        # initialize KFold
        shuffle = True

        # use EqualGroupsKFold with clustering regression (to have equal benchmarks from each cluster in the testing/training)
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

            result = [[fold for i in range(errors.shape[0])],
                      test_y, pred_y, errors]
            result_columns = ["fold", "Real " + str(target_name[0]), "Predicted " + str(target_name[0]),
                              "Average Error (%)"]

            if with_clustering:
                clusters = estimator.clusters
                features = estimator.features
                predicted_estimators = estimator.estimators_pred
                result.append(clusters)
                result.append(predicted_estimators)
                result_columns += ["Cluster", "Estimator"]

            # try to get the estimator coefs if it is Linear or Exponential
            try:
                a = estimator.coef_
                b = estimator.intercept_
                equations = [get_equation(a[i], b[i], features[i]) for i in range(errors.shape[0])]
                result.append(equations)
                result_columns += ["Equation"]
            except AttributeError:
                a = None

            if not with_clustering:
                features = list(features_names)
                result = np.array(result).transpose().tolist()
                for g in range(len(result)):
                    result[g] = test_x_before_scaling[g].tolist()\
                                + list(result[g])
                result_columns = features + result_columns
            else:
                result = np.array(result).transpose().tolist()

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

        return {"results": results_df, "stats": stats_df, "training_error": avg_training_error, "testing_error": np.mean(all_errors)}


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


    # ret is the output from validate_model
    def store_output(self,ret, sheet_name=None, excelname1=None, excelname2=None, mode='w', with_clustering=False):

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
            if with_clustering:
                avg_n_runs = pd.concat(ret["results"][c])[
                    ["Real " + str(self.target_name[0]), "Predicted " + str(self.target_name[0]),
                     "Average Error (%)", "Cluster"]].groupby(
                    level=0).mean()
                
            else:
                avg_n_runs = pd.concat(ret["results"][c])[["Real " + str(self.target_name[0]), "Predicted " + str(self.target_name[0]),"Average Error (%)"]].groupby(
                level=0).mean()

            std_n_runs = \
            pd.concat(ret["results"][c])[["Real " + str(self.target_name[0]), "Predicted " + str(self.target_name[0]),
                 "Average Error (%)"]].groupby(level=0).std()
            std_n_runs = std_n_runs.rename(columns={"Average Error (%)": "STD of Error"})
            summary_n_runs = pd.concat([avg_n_runs, std_n_runs["STD of Error"]], axis=1)
            stats_n_runs = pd.concat(ret["stats"][c]).groupby(level=0).mean()
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
                error_4each_cluster.append(np.mean(ret["avg_testing_error"][c]))
                error_label_4each_cluster.append("Cluster {} Error (%)".format(c))
                feature_label_4each_cluster.append("Cluster {} Features".format(c))
                estimators_label_4each_cluster.append("Cluster {} Estimator".format(c))
        results_n_runs = pd.concat(summary_4each_cluster)

        # stats has the overall avg error, average stdev of errors, and overall RMSE
        stats = pd.concat(stats_4each_cluster).groupby(level=0).mean()
        features = pd.DataFrame([ret["features_4each_cluster"][:self.n_clusters]], columns=feature_label_4each_cluster[:self.n_clusters])
        estimators = pd.DataFrame([ret["estimator_4each_cluster"][:self.n_clusters]], columns=estimators_label_4each_cluster[:self.n_clusters])
        errors = pd.DataFrame([error_4each_cluster], columns=error_label_4each_cluster)

        row = 0
        # store the results_n_runs first

        column = 0

        results_n_runs.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column)
        results_n_runs.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column)
        row = row + len(results_n_runs.index.tolist()) + 2

        estimators.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column)
        estimators.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column)
        column += len(estimators.columns.tolist())+2
        features.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column)
        features.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column)
        column += len(features.columns.tolist())+2
        errors.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column)
        errors.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column)
        column += len(features.columns.tolist()) + 2
        stats.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column)
        stats.to_excel(writer2, startrow=row, sheet_name=sheet_name, startcol=column)
        column = 0
        row = row + 3

        for j in range(self.n_runs):
            if j > 10:
                break
            for c in range(n_clusters):
                result = ret["results"][c][j]
                result.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column, index_label = "Cluster {}".format(c))
                row = row + len(result.index.tolist()) + 2
                stat = ret["stats"][c][j]
                stat.to_excel(writer1, startrow=row, sheet_name=sheet_name, startcol=column, index_label = "Cluster {}".format(c))
                row = row + len(stat.index.tolist()) + 2
            row = row + 1

        writer1.save()
        writer2.save()
        print("FPGAPredict: output file was saved as \"%s\"" %result_name1)

        return result_name1, result_name2

    def print_result(self, ret, with_clustering=False):
        if with_clustering:
            n_clusters = 1
        else:
            n_clusters = self.n_clusters
        for j in range(self.n_runs):
            if j > 10:
                break
            for c in range(n_clusters):
                print("cluster {}:\n".format(c))
                result = ret["results"][c][j]
                print(result.to_string())
                stat = ret["stats"][c][j]
                print(stat.to_string())
                print("\n")

            print("\n")

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
