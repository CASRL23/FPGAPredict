from estimators import *
import sys
from ML import FPGAPredict
import warnings

warnings.filterwarnings("ignore")

import json
from utils import *

if __name__ == "__main__":

    # load configurations
    # command line arguments are enough to pass the required parameters and will overwrite any configurations in the config.json file.
    f = open("config.json")
    params = json.load(f)

    # adjust configurations based on command line inputs (used for quick adjustments instead of editing the file)
    cmd_args = sys.argv[1:]

    for arg in cmd_args:
        if arg == "--usage":
            usage()
            exit()
        arg = arg.split("=")
        param = arg[0].replace(" ", "")  # remove space
        input = arg[1]
        try:

            params[param] = input

        except KeyError:
            print("%s is not defined." % param)
            usage()
            exit()

    # make the required conversions on inputs and validate them.
    params = validate_inputs(params)
    print("\n*** main.py: Running with the following configuration: \n%s\n" % params)

    params["shuffle"] = True

    # initialize FPGAPredict with the configuration
    fpga_pred = FPGAPredict(params)

    # read the data set from excel
    x_input, y_target, x_clustering = fpga_pred.read_excel_sheet(features=params["features"],
                                                                 target=[params["target_metric"]],
                                                                 clustering=params["clustering_features"])

    # get the total number of benchmarks
    n_benchmarks = len(fpga_pred.benchmarks)

    # select the top clustering features (only if n_clusters > 1 and clustering_perm=True)
    top_clustering_features = fpga_pred.select_clustering_features(x_input, y_target, x_clustering,
                                                                   n=params["clustering_n"],
                                                                   ex_search=params["clustering_ex_search"],
                                                                   min_features=params["clustering_min_features"],
                                                                   max_features=params["clustering_max_features"])

    # prepare an array to store the average error of each top clustering feature subset
    avg_error_4each_clustering_subset = []
    output_4each_clustering_subset = []
    result_4each_clustering_subset = []

    # iterate through the top 2 clustering features
    for (count, clustering_features) in enumerate(top_clustering_features):
        if count > 1:  # only go over the first two top clustering features
            break
        print("\n-----------------------------------------------------\n**main.py: Running an experiment with the "
              "clustering features %s\n-----------------------------------------------------\n" %clustering_features)
        # select the prediction features for each cluster
        features_4each_cluster = fpga_pred.select_prediction_features(x_input, y_target, x_clustering[clustering_features],
                                                                      estimator_4each_cluster=params[
                                                                          "estimator_4each_cluster"],
                                                                      ex_search=params["prediction_ex_search"],
                                                                      n = params["prediction_n"],
                                                                      min_features=params["prediction_min_features"],
                                                                      max_features=params["prediction_max_features"])

        # find x and y for each cluste rusing get_clusters and pass them to validate_model
        # validate the model with the seeded shuffle
        print("\n**main.py: evaluating the model for each cluster separately and with seeded shuffle\n")
        ret = fpga_pred.evaluate_model(x_input, y_target, x_clustering[clustering_features],
                                       estimator_4each_cluster=params["estimator_4each_cluster"],
                                       features_4each_cluster=features_4each_cluster,
                                       with_clustering=False,
                                       random_state=params["random_state"])

        # print 10 of the results
        fpga_pred.print_result(ret)

        # store the output in excel file
        excelname1, excelname2, results_n_runs = fpga_pred.store_output(ret, sheet_name="seeded_{}_runs".format(params["n_runs"]))
        ##############################################################
        # now validate the model with true shuffle
        print("\n**main.py: evaluating the model for each cluster separately and with true shuffle\n")
        ret = fpga_pred.evaluate_model(x_input, y_target, x_clustering[clustering_features],
                                       estimator_4each_cluster=params["estimator_4each_cluster"],
                                       features_4each_cluster=features_4each_cluster,
                                       with_clustering=False,
                                       random_state=None)
        # print 10 of the results
        fpga_pred.print_result(ret)

        # store the results in another sheet
        excelname1, excelname2, results_n_runs = fpga_pred.store_output(ret, sheet_name="true_{}_runs".format(
            fpga_pred.n_runs),
                                                                        excelname1=excelname1, excelname2=excelname2,
                                                                        mode='a')

        ############################################################################
        # now validate the model with seeded shuffle and with clustering
        print("\n**main.py: evaluating the model with clustering and prediction together and with seeded shuffle\n")
        ret = fpga_pred.evaluate_model(x_input, y_target, x_clustering[clustering_features],
                                       estimator_4each_cluster=params["estimator_4each_cluster"],
                                       features_4each_cluster=features_4each_cluster,
                                       with_clustering=True,
                                       random_state=params["random_state"])
        avg_error_4each_clustering_subset.append(ret["overall_avg_error"])
        output_4each_clustering_subset.append(excelname1)
        # print 10 of the results
        fpga_pred.print_result(ret, with_clustering=True)

        # store the results as a new sheet
        excelname1, excelname2, results_n_runs = fpga_pred.store_output(ret,
                                                                        sheet_name="seeded_{}runs_clustering".format(
                                                                            fpga_pred.n_runs),
                                                                        excelname1=excelname1,
                                                                        excelname2=excelname2, mode='a',
                                                                        with_clustering=True)
        result_4each_clustering_subset.append(results_n_runs)

        # now validate the model with true shuffle and with clustering
        print("\n**main.py: evaluating the model with clustering and prediction together and with true shuffle\n")
        ret = fpga_pred.evaluate_model(x_input, y_target, x_clustering[clustering_features],
                                       estimator_4each_cluster=params["estimator_4each_cluster"],
                                       features_4each_cluster=features_4each_cluster,
                                       with_clustering=True,
                                       random_state=None)

        # print 10 of the results
        fpga_pred.print_result(ret, with_clustering=True)

        # store the results in a new sheet
        excelname1, excelname2, results_n_runs = fpga_pred.store_output(ret, sheet_name="true_{}runs_clustering".format(
            fpga_pred.n_runs),
                                                                        excelname1=excelname1,
                                                                        excelname2=excelname2, mode='a',
                                                                        with_clustering=True)
    # find the best clustering features from the top 2
    argmin = np.argmin(avg_error_4each_clustering_subset)
    best_result = output_4each_clustering_subset[argmin]
    best_clustering_subset = top_clustering_features[argmin]
    test_df = result_4each_clustering_subset[argmin]

    print("\n**main.py: The best clustering features are %s with an average error of %.2f.\nThis result was stored as "
          "%s\n" % (best_clustering_subset, avg_error_4each_clustering_subset[argmin], best_result))

    # if this is a regression test then check if the results match
    if params["regression_test"] != "None":
        real_file = params["regression_test"]
        try:
            real_df = pd.read_excel(real_file, sheet_name="seeded_" + str(fpga_pred.n_runs) + "runs_clustering",
                                    engine="openpyxl", index_col=0, header=0, nrows=n_benchmarks)
        except (KeyError, ValueError) as e1:
            try:
                real_df = pd.read_excel(real_file, sheet_name="Summary of " + str(fpga_pred.n_runs) + " runs",
                                        engine="openpyxl", index_col=0, header=0, nrows=n_benchmarks)
            except (KeyError, ValueError) as e2:
                print("**main.py: Error: regression test file is not valid.\n")
                print("Error Message: %s" % e2)
                exit(1)
        real_df = real_df.dropna(axis=1)
        real_df.index.name = None
        real_df = real_df.astype("float")
        test_df = test_df.astype("float")
        real_df = real_df.round(decimals=2)
        test_df = test_df.round(decimals=2)

        if test_df.equals(real_df):
            print("\n--------------------\nRegression Test PASSED!\n")
            # plot all results
            fpga_pred.plot_show()
            exit(0)
        else:
            print("\n--------------------\nRegression Test FAILED!\n")
            print("\nThe correct output dataframe is: \n\n%s\n\nBut the current outptu dataframe is: \n\n%s\n" % (
                real_df.to_string(), test_df.to_string()))
            diff = pd.concat([real_df, test_df]).drop_duplicates(keep=False)
            print("\nThe difference between the two dataframes are in the following:\n\n%s\n\n" % (diff.to_string()))
            # plot all results
            fpga_pred.plot_show()
            exit(1)
