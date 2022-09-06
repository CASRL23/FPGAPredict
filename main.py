from estimators import *
import sys
from ML import FPGAPredict
import warnings

warnings.filterwarnings("ignore")

import json
from utils import *

if __name__ == "__main__":

    # load configurations
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
    print("Running with the following configuration: \n%s\n" % inputs)

    # initialize FPGAPredict with the configuration
    fpga_pred = FPGAPredict(excelname=str(inputs["excelname"]),
                            estimator=inputs["estimator"],
                            n=int(inputs["n"]),
                            perm=inputs["perm"],
                            clustering_n=inputs["clustering_n"],
                            clustering_estimators=inputs["clustering_estimators"],
                            clusters_features=inputs["clusters_features"],
                            clustering_perm=inputs["clustering_perm"],
                            random_state=inputs["random_state"],
                            n_clusters=inputs["n_clusters"],
                            shuffle=False,
                            n_splits=inputs["n_splits"],
                            printt=inputs["print"],
                            scaling=inputs["scaling"],
                            n_runs=inputs["n_runs"],
                            perm_min=inputs["perm_min"],
                            clustering_perm_min=inputs["clustering_perm_min"],
                            perm_up_to=inputs["perm_up_to"],
                            clustering_perm_up_to=inputs["clustering_perm_up_to"],
                            parallel=inputs["parallel"], n_cpu=inputs["n_cpu"])

    # start running FPGAPredict
    ret = fpga_pred.run(target=[inputs["target_feature"]], features=inputs["features"],
                        clustering_features=inputs["clustering_features"])

    # save the first and second top clustering features
    first_clustering_features = fpga_pred.clustering_features
    second_clustering_features = fpga_pred.second_clustering_features

    # get the average error
    avg_error = fpga_pred.get_avg_error()

    excelname1, excelname2, features = fpga_pred.store_output(ret)

    fpga_pred.clusters_features = features
    fpga_pred.perm=False
    fpga_pred.clustering_perm=False

    # turn on true shuffle, random state will be set to None automatically
    fpga_pred.shuffle = True

    ret = fpga_pred.run(target=[inputs["target_feature"]], features=inputs["features"],
                        clustering_features=fpga_pred.clustering_features)
    fpga_pred.store_output(ret,excelname1=excelname1,excelname2=excelname2,sheet_name="true_" + str(fpga_pred.n_runs) + "_runs", mode='a')

    if inputs["n_clusters"] > 1:
        fpga_pred.estimator = ESTIMATORS[0]
        fpga_pred.shuffle = False
        fpga_pred.estimator.set_parameters(
            estimators=list(zip([ESTIMATORS[inputs["estimator"]] for i in range(fpga_pred.n_clusters)], features)))

        ret = fpga_pred.run(target=[inputs["target_feature"]], features=inputs["features"],
                            clustering_features=fpga_pred.clustering_features)
        avg_error = fpga_pred.get_avg_error()

        fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2,
                               sheet_name="seeded_" + str(fpga_pred.n_runs) + "runs_clustering", mode='a')

        # turn on true shuffle
        fpga_pred.shuffle = True
        ret = fpga_pred.run(target=[inputs["target_feature"]], features=inputs["features"],
                            clustering_features=fpga_pred.clustering_features)

        fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2,
                               sheet_name="true_" + str(fpga_pred.n_runs) + "runs_clustering", mode='a')

    else:
        second_clustering_features = None

    first_excelname1 = excelname1
    first_excelname2 = excelname2
    avg_error2 = np.Inf

    if second_clustering_features is not None:
        fpga_pred.clustering_features = second_clustering_features
        fpga_pred.perm = inputs["perm"]
        fpga_pred.shuffle = False
        fpga_pred.estimator = ESTIMATORS[inputs["estimator"]]
        fpga_pred.clusters_features = None
        # run with the second top clustering features
        ret = fpga_pred.run(target=[inputs["target_feature"]], features=inputs["features"],
                            clustering_features=fpga_pred.clustering_features)
        excelname1, excelname2, features = fpga_pred.store_output(ret)

        fpga_pred.clusters_features = features
        fpga_pred.perm = False
        fpga_pred.clustering_perm = False

        # turn on true shuffle, random state will be set to None automatically
        fpga_pred.shuffle = True

        ret = fpga_pred.run(target=[inputs["target_feature"]], features=inputs["features"],
                            clustering_features=fpga_pred.clustering_features)
        fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2, sheet_name="true_" + str(fpga_pred.n_runs) + "_runs", mode='a')

        if inputs["n_clusters"] > 1:

            fpga_pred.estimator = ESTIMATORS[0]
            fpga_pred.shuffle = False
            fpga_pred.estimator.set_parameters(
                estimators=list(zip([ESTIMATORS[inputs["estimator"]] for i in range(fpga_pred.n_clusters)], features)))

            ret = fpga_pred.run(target=[inputs["target_feature"]], features=inputs["features"],
                                clustering_features=fpga_pred.clustering_features)
            avg_error2 = fpga_pred.get_avg_error()
            fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2,
                                sheet_name="seeded_" + str(fpga_pred.n_runs) + "runs_clustering", mode='a')

            # turn on true shuffle
            fpga_pred.shuffle = True
            ret = fpga_pred.run(target=[inputs["target_feature"]], features=inputs["features"],
                            clustering_features=fpga_pred.clustering_features)
            fpga_pred.store_output(ret, excelname1=excelname1, excelname2=excelname2,
                               sheet_name="true_" + str(fpga_pred.n_runs) + "runs_clustering", mode='a')
            second_excelname1 = excelname1
            second_excelname2 = excelname2
    print()
    if avg_error2 < avg_error:
        print("The best clustering features are: %s with an average error of %.2f %%" %(second_clustering_features, avg_error2))
        print("The output file was saved as %s" %second_excelname1)
    else:
        print("The best clustering features are %s with an average error of %.2f %%" %(first_clustering_features, avg_error))
        print("The output file was saved as %s" %first_excelname1)