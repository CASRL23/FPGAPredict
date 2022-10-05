from estimators import ESTIMATORS
import os
from distutils.util import strtobool

'''
A function that processes the inputs and validate them. If not valid inputs, an error is thrown.
'''


def validate_inputs(inputs):
    inputs["n_runs"] = int(inputs["n_runs"])
    inputs["scaling"] = int(inputs["scaling"])
    try:
        inputs["estimator"] = int(inputs["estimator"])
    except ValueError:
        inputs["estimator"] = str(inputs["estimator"])

    inputs["prediction_min_features"] = int(inputs["prediction_min_features"])
    inputs["clustering_min_features"] = int(inputs["clustering_min_features"])
    inputs["prediction_ex_search"] = strtobool(inputs["prediction_ex_search"])
    inputs["clustering_ex_search"] = strtobool(inputs["clustering_ex_search"])
    inputs["clustering_n"] = int(inputs["clustering_n"])
    inputs["n_clusters"] = int(inputs["n_clusters"])
    inputs["n_splits"] = int(inputs["n_splits"])
    inputs["print"] = strtobool(inputs["print"])
    inputs["parallel"] = strtobool(inputs["parallel"])
    inputs["prediction_n"] = int(inputs["prediction_n"])
    if inputs["estimator_4each_cluster"] == "None" or inputs["estimator_4each_cluster"] is None:
        inputs["estimator_4each_cluster"] = None
    else:
        if isinstance(inputs["estimator_4each_cluster"], str):
            inputs["estimator_4each_cluster"] = inputs["estimator_4each_cluster"].split(",")
    if isinstance(inputs["estimator_4each_cluster"], list):
        for a in range(len(inputs["estimator_4each_cluster"])):
            inputs["estimator_4each_cluster"][a] = ESTIMATORS[inputs["estimator_4each_cluster"][a]]

    if inputs["n_cpu"] == "None" or inputs["n_cpu"] == None:
        inputs["n_cpu"] = None
    else:
        inputs["n_cpu"] = int(inputs["n_cpu"])

    if inputs["prediction_max_features"] == "None" or inputs["prediction_max_features"] == None:
        inputs["prediction_max_features"] = None
    else:
        inputs["prediction_max_features"] = int(inputs["prediction_max_features"])

    if inputs["clustering_max_features"] == "None" or inputs["clustering_max_features"] == None:
        inputs["clustering_max_features"] = None
    else:
        inputs["clustering_max_features"] = int(inputs["clustering_max_features"])
    if inputs["features"] == "None" or inputs["features"] == None:
        inputs["features"] = None

    else:
        inputs["features"] = list(inputs["features"].split(","))

    if inputs["target_metric"] == "None" or inputs["target_metric"] == None:
        inputs["target_metric"] = None
    else:
        inputs["target_metric"] = str(inputs["target_metric"])

    if inputs["clustering_features"] == "None" or inputs["clustering_features"] == None:
        inputs["clustering_features"] = None
    elif isinstance(inputs["clustering_features"], str):
        inputs["clustering_features"] = list(inputs["clustering_features"].split(","))
    if inputs["random_state"] == "None" or inputs["random_state"] == None:
        inputs["random_state"] = None
    else:
        inputs["random_state"] = int(inputs["random_state"])

    # start validating inputs

    if inputs["features"] is not None and not isinstance(inputs["features"], list):
        raise ValueError("FPGAPredict: The features passed must be a list or None. Please read the readme file for "
                         "more information.")
        exit(1)
    if inputs["clustering_features"] is not None and not isinstance(inputs["clustering_features"], list):
        raise ValueError(
            "FPGAPredict: The clustering features passed must be a list or None. Please read the readme file for more "
            "information.")
        exit(1)
    if isinstance(inputs["estimator"], int):
        if inputs["estimator"] < 0 or inputs["estimator"] >= len(ESTIMATORS):
            raise ValueError(
                "FPGAPredict: The estimator must be an integer between the values %d and %d from the following dictionary: \n%s\nPlease read the readme file for more information." % (
                0, len(ESTIMATORS) - 1, ESTIMATORS))

    if inputs["estimator"] == 0 and (
            inputs["estimators_4each_cluster"] is None or np.array(inputs["estimators_4each_cluster"]).shape[0] >=
            inputs["n_clusters"]):
        raise ValueError(
            "FPGAPredict: clustering estimators or the prediction features for each cluster were not set. Please set them and try again.")
    if inputs["n_clusters"] < 1:
        raise ValueError("FPGAPredict: the number of clusters must be 1 or more.")
    if inputs["n_splits"] < 1:
        raise ValueError("FPGAPredict: the number of KFold splits (n_splits) must be 1 or more.")
    inputs["estimator"] = ESTIMATORS[inputs["estimator"]]

    return inputs


'''
This function just prints the usage of FPGAPredict. It basically prints the readme file.
'''


def usage():
    f = open("readme", "r")
    for line in f.readlines():
        print(line)
    f.close()


'''
This function takes as input a file name and a directory.
It then checks if this file name is already taken in the directory,
if it is taken, it adds an appropriate number to the file name so that it won't override the old file.
It finally returns the full path to the file name.
'''


def name_file(_name, path, extension):
    # prepare a file to store all feature subsets that give an error < 5%
    all_found_features = []
    i = 0
    _name = _name + str(i)
    # _name = "perm_results"
    try:
        files = os.listdir(path)

        while _name + extension in files:
            i += 1
            files.remove(_name + extension)
            j = 0
            while _name[j - 1].isdigit():
                j -= 1
            if j != 0:

                _name = _name[:j] + str(i)
            else:
                _name = _name + str(i)

        _name = path + _name + extension
        return _name
    except FileNotFoundError:
        os.mkdir(path)

    return path + _name + extension
