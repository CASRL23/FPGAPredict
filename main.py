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
    inputs = json.load(f)

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

            inputs[param] = input

        except KeyError:
            print("%s is not defined." % param)
            usage()
            exit()

    # make the required conversions on inputs and validate them.
    inputs = validate_inputs(inputs)
    print("\n*** main.py: Running with the following configuration: \n%s\n" % inputs)

    inputs["shuffle"] = True

    # initialize FPGAPredict with the configuration
    fpga_pred = FPGAPredict(inputs)

    # start running FPGAPredict
    print("\n*** main.py: Running FPGAPredict for each cluster separately to select the features..\n")
    ret = fpga_pred.run(target=[inputs["target_metric"]], features=inputs["features"],
                        clustering_features=inputs["clustering_features"])
    n_benchmarks = len(fpga_pred.benchmarks)

    # save the first and second top clustering features
    if len(fpga_pred.top_clustering_features) > 0:
        first_clustering_features = fpga_pred.top_clustering_features[0]
    else:
        first_clustering_features = inputs["clustering_features"]
    if len(fpga_pred.top_clustering_features) > 1:
        second_clustering_features = fpga_pred.top_clustering_features[1]
    else:
        second_clustering_features = None

    # get the average error
    avg_error = fpga_pred.get_avg_error()

    excelname1, excelname2, features = fpga_pred.store_output(ret)

    fpga_pred.clusters_features = features
    fpga_pred.perm = False
    fpga_pred.clustering_perm = False

    # turn on true shuffle, random state will be set to None automatically
    fpga_pred.true_shuffle = True

    print("\n *** main.py: Running FPGAPredict with each cluster separately with true shuffle..\n")
    ret = fpga_pred.run(target=[inputs["target_metric"]], features=inputs["features"],
                        clustering_features=fpga_pred.clustering_features)
    fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2,
                           sheet_name="true_" + str(fpga_pred.n_runs) + "_runs", mode='a')

    if inputs["n_clusters"] > 1:

        fpga_pred.estimator = ESTIMATORS[0]
        fpga_pred.true_shuffle = False
        fpga_pred.estimator.set_parameters(
            estimators=list(zip([ESTIMATORS[inputs["estimator"]] for i in range(fpga_pred.n_clusters)], features)))
        print("\n *** main.py: Running FPGAPredict with clustering regression and with seeded shuffle...\n")
        ret = fpga_pred.run(target=[inputs["target_metric"]], features=inputs["features"],
                            clustering_features=fpga_pred.clustering_features)
        avg_error = fpga_pred.get_avg_error()

        fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2,
                               sheet_name="seeded_" + str(fpga_pred.n_runs) + "runs_clustering", mode='a')
        test_df1 = fpga_pred.results
        # turn on true shuffle
        fpga_pred.true_shuffle = True
        print("*** main.py: Running FPGAPredict with clustering regression and with true shuffle...\n")
        ret = fpga_pred.run(target=[inputs["target_metric"]], features=inputs["features"],
                            clustering_features=fpga_pred.clustering_features)

        fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2,
                               sheet_name="true_" + str(fpga_pred.n_runs) + "runs_clustering", mode='a')

    else:
        second_clustering_features = None

    first_excelname1 = excelname1
    first_excelname2 = excelname2
    avg_error2 = np.Inf

    if second_clustering_features is not None:
        print("\n------------------------------------------------------------\n")
        print("\n*** main.py: Running FPGAPredict with the second top clustering features: %s ...\n\n" %second_clustering_features)

        fpga_pred.clustering_features = second_clustering_features
        fpga_pred.perm = inputs["perm"]
        fpga_pred.true_shuffle = False
        fpga_pred.estimator = ESTIMATORS[inputs["estimator"]]
        fpga_pred.clusters_features = None

        print("\n*** main.py: Running FPGAPredict for each cluster separately to select the features..\n")
        # run with the second top clustering features
        ret = fpga_pred.run(target=[inputs["target_metric"]], features=inputs["features"],
                            clustering_features=fpga_pred.clustering_features)
        excelname1, excelname2, features = fpga_pred.store_output(ret)

        fpga_pred.clusters_features = features
        fpga_pred.perm = False
        fpga_pred.clustering_perm = False

        # turn on true shuffle, random state will be set to None automatically
        fpga_pred.true_shuffle = True

        print("\n *** main.py: Running FPGAPredict with each cluster separately with true shuffle..\n")
        ret = fpga_pred.run(target=[inputs["target_metric"]], features=inputs["features"],
                            clustering_features=fpga_pred.clustering_features)
        fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2,
                               sheet_name="true_" + str(fpga_pred.n_runs) + "_runs", mode='a')

        if inputs["n_clusters"] > 1:
            fpga_pred.estimator = ESTIMATORS[0]
            fpga_pred.true_shuffle = False
            fpga_pred.estimator.set_parameters(
                estimators=list(zip([ESTIMATORS[inputs["estimator"]] for i in range(fpga_pred.n_clusters)], features)))

            print("\n *** main.py: Running FPGAPredict with clustering regression and with seeded shuffle..\n")
            ret = fpga_pred.run(target=[inputs["target_metric"]], features=inputs["features"],
                                clustering_features=fpga_pred.clustering_features)
            avg_error2 = fpga_pred.get_avg_error()
            fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2,
                                   sheet_name="seeded_" + str(fpga_pred.n_runs) + "runs_clustering", mode='a')

            test_df2 = fpga_pred.results
            # turn on true shuffle
            fpga_pred.true_shuffle = True
            print("\n *** main.py: Running FPGAPredict with clustering regression and with true shuffle..\n")
            ret = fpga_pred.run(target=[inputs["target_metric"]], features=inputs["features"],
                                clustering_features=fpga_pred.clustering_features)
            fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2,
                                   sheet_name="true_" + str(fpga_pred.n_runs) + "runs_clustering", mode='a')
            second_excelname1 = excelname1
            second_excelname2 = excelname2

    if avg_error2 < avg_error and inputs["n_clusters"] > 1:
        print("\nThe best clustering features are: %s with an average error of %.2f %%" % (
            second_clustering_features, avg_error2))
        print("The output file was saved as %s" % second_excelname1)
        test_df = test_df2
    elif inputs["n_clusters"] > 1:
        print("\nThe best clustering features are %s with an average error of %.2f %%" % (
            first_clustering_features, avg_error))
        print("The output file was saved as %s" % first_excelname1)
        test_df = test_df1# check if we are doing regression test
    if inputs["regression_test"] != "None":
        real_file = inputs["regression_test"]
        real_df = pd.read_excel(real_file, sheet_name="seeded_" + str(fpga_pred.n_runs) + "runs_clustering", engine="openpyxl", index_col=0, header=0, nrows=n_benchmarks)
        real_df = real_df.dropna(axis=1)
        real_df.index.name = None
        real_df = real_df.astype("float")
        test_df = test_df.astype("float")
        real_df = real_df.round(decimals=2)
        test_df = test_df.round(decimals=2)

        if test_df.equals(real_df):
            print("\n--------------------\nRegression Test PASSED!\n")
        else:
            print("\n--------------------\nRegression Test FAILED!\n")
            print("\nThe correct output dataframe is: \n\n%s\n\nBut the current outptu dataframe is: \n\n%s\n" %(real_df.to_string(), test_df.to_string()))
            diff = pd.concat([real_df, test_df]).drop_duplicates(keep=False)
            print("\nThe difference between the two dataframes are in the following:\n\n%s\n\n" %(diff.to_string()))
