from estimators import *
import sys
from new_ML import FPGAPredict
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

    # read the data set from excel
    x_input, y_target, x_clustering = fpga_pred.read_excel_sheet(features=inputs["features"], target=[inputs["target_metric"]], clustering=inputs["clustering_features"])

    # select the top clustering features (only if n_clusters > 1 and clustering_perm=True)
    top_clustering_features = fpga_pred.select_clustering_features(x_input, y_target, x_clustering)

    # prepare an array to store the average error of each top clustering feature subset
    avg_error_4each_clustering_subset = []
    output_4each_clustering_subset = []

    # iterate through the top 2 clustering features
    for count,feature in enumerate(top_clustering_features):
        if count > 1: # only go over the first two top clustering features
            break
        # select the prediction features for each cluster
        features_4each_cluster = fpga_pred.select_prediction_features(x_input, y_target, x_clustering,                                                                      clustering_features=feature)

        # validate the model with the seeded shuffle
        fpga_pred.random_state = inputs["random_state"]
        ret = fpga_pred.validate_model(x_input, y_target, x_clustering[feature],
                                       estimators_4each_cluster=inputs["estimators"],
                                       features_4each_cluster=features_4each_cluster,
                                       with_clustering=False,
                                       random_state=inputs["random_state"], true_shuffle=False)

        # print 10 of the results
        fpga_pred.print_result(ret)

        # store the output in excel file
        excelname1, excelname2 = fpga_pred.store_output(ret)
        ##############################################################
        # now validate the model with true shuffle
        ret = fpga_pred.validate_model(x_input, y_target, x_clustering[feature],
                                       estimators_4each_cluster=inputs["estimators"],
                                       features_4each_cluster=features_4each_cluster,
                                       with_clustering=False,
                                       true_shuffle=True,
                                       random_state=None)
        # print 10 of the results
        fpga_pred.print_result(ret)

        # store the results in another sheet
        fpga_pred.store_output(ret, sheet_name="true_{}_runs".format(fpga_pred.n_runs),
                               excelname1=excelname1, excelname2=excelname2, mode='a')

        ############################################################################
        # now validate the model with seeded shuffle and with clustering
        ret = fpga_pred.validate_model(x_input, y_target, x_clustering[feature],
                                       estimators_4each_cluster=inputs["estimators"],
                                       features_4each_cluster=features_4each_cluster,
                                       with_clustering=True,
                                       true_shuffle=False, random_state=inputs["random_state"])
        avg_error_4each_clustering_subset.append(ret["overall_avg_error"])
        output_4each_clustering_subset.append(excelname1)
        # print 10 of the results
        fpga_pred.print_result(ret, with_clustering=True)

        # store the results as a new sheet
        fpga_pred.store_output(ret, sheet_name="seeded_{}runs_clustering".format(fpga_pred.n_runs),
                               excelname1=excelname1,
                               excelname2=excelname2, mode='a', with_clustering=True)

        # now validate the model with true shuffle and with clustering
        ret = fpga_pred.validate_model(x_input, y_target, x_clustering[feature],
                                       estimators_4each_cluster=inputs["estimators"],
                                       features_4each_cluster=features_4each_cluster,
                                       with_clustering=True,
                                       true_shuffle=True, random_state=None)

        # print 10 of the results
        fpga_pred.print_result(ret, with_clustering=True)

        # store the results in a new sheet
        fpga_pred.store_output(ret, sheet_name="true_{}runs_clustering".format(fpga_pred.n_runs),
                               excelname1=excelname1,
                               excelname2=excelname2, mode='a', with_clustering=True)
    argmin = np.argmin(avg_error_4each_clustering_subset)
    best_result = output_4each_clustering_subset[argmin]
    best_clustering_subset = top_clustering_features[argmin]

    print("\n**main.py: The best clustering features are %s with an average error of %.2f.\nThis result was stored as "
          "%s\n" %(best_clustering_subset, avg_error_4each_clustering_subset[argmin],best_result))
