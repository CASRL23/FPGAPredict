import os
import sys

params = {"tests": []}

# adjust configurations based on command line inputs (used for quick adjustments instead of editing the file)
cmd_args = sys.argv[1:]
for arg in cmd_args:
    arg = arg.split("=")
    param = arg[0].replace(" ", "")  # remove space
    input = arg[1]
    tests = str(input).split(",")
    for a in range(len(tests)):
        tests[a] = int(tests[a])
    params[param] = tests
if len(params["tests"]) == 0:
    params["tests"] = [i for i in range(7)]
total_exit_codes = 0
total_tests = 0

# test1: run with the clock period, 2 clusters, and linear regression. Permutation with 10 features only.
if 1 in params["tests"]:
    exit_code = os.system(
        "python3 main.py estimator=1 prediction_n=10 prediction_ex_search=True excelname=arria10_features.xlsx clustering_n=10 clustering_ex_search=True n_clusters=2 features=None target_metric=\"Clock Period\" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 clustering_min_features=1 prediction_min_features=2 prediction_max_features=None clustering_max_features=5 n_cpu=None parallel=True print=False regression_test=./test/test1.xlsx")
    total_exit_codes += exit_code
    total_tests += 1

if 2 in params["tests"]:
    # test2: run with the clock period, 2 clusters, and exponential regression. Permutation with 10 features only.
    exit_code = os.system(
        "python3 main.py estimator=2 prediction_n=10 prediction_ex_search=True excelname=arria10_features.xlsx clustering_n=10 clustering_ex_search=True n_clusters=2 features=None target_metric=\"Clock Period\" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 clustering_min_features=1 prediction_min_features=2 prediction_max_features=None clustering_max_features=5 n_cpu=None parallel=True print=False regression_test=./test/test2.xlsx")
    total_exit_codes += exit_code
    total_tests += 1

if 3 in params["tests"]:
    # test3: run with the clock period, 2 clusters, and random forest. Permutation with 10 features only.
    exit_code = os.system(
        "python3 main.py estimator="RF" prediction_n=10 prediction_ex_search=True excelname=arria10_features.xlsx clustering_n=10 clustering_ex_search=True n_clusters=2 features=None target_metric=\"Clock Period\" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 clustering_min_features=1 prediction_min_features=2 prediction_max_features=None clustering_max_features=5 n_cpu=None parallel=True print=False regression_test=./test/test3.xlsx")
    total_exit_codes += exit_code
    total_tests += 1

if 4 in params["tests"]:
    # test4: run with the clock period, 3 clusters, and linear regression. Permutation with 5 features only.
    exit_code = os.system(
        "python3 main.py estimator=1 prediction_n=5 prediction_ex_search=True excelname=arria10_features.xlsx clustering_n=10 clustering_ex_search=True n_clusters=3 features=None target_metric=\"Clock Period\" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 clustering_min_features=1 prediction_min_features=2 prediction_max_features=None clustering_max_features=5 n_cpu=None parallel=True print=False regression_test=./test/test4.xlsx")
    total_exit_codes += exit_code
    total_tests += 1

if 5 in params["tests"]:
    # test5: run with the clock cycles, 3 clusters, and linear regression. Permutation with 5 features only.
    exit_code = os.system(
        "python3 main.py estimator=1 prediction_n=5 prediction_ex_search=True excelname=arria10_features.xlsx clustering_n=10 clustering_ex_search=True n_clusters=3 features=None target_metric=\"Clock Cycles\" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 clustering_min_features=1 prediction_min_features=2 prediction_max_features=None clustering_max_features=5 n_cpu=None parallel=True print=False regression_test=./test/test6.xlsx")
    total_exit_codes += exit_code
    total_tests += 1

if 6 in params["tests"]:
    # test7: run with the clock cycles, 3 clusters, and exponential regression. Permutation with 10 features only.
    exit_code = os.system(
        "python3 main.py estimator=2 prediction_n=10 prediction_ex_search=True excelname=arria10_features.xlsx clustering_n=10 clustering_ex_search=True n_clusters=3 features=None target_metric=\"Clock Cycles\" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 clustering_min_features=1 prediction_min_features=2 prediction_max_features=None clustering_max_features=5 n_cpu=None parallel=True print=False regression_test=./test/test7.xlsx")
    total_exit_codes += exit_code
    total_tests += 1

# test8: run with the clock cycles, 1 cluster, and ANN. Permutation with 5 features only.
# (NOTE: this test results may change if you change ANN configurations in estimators.py)
# ANN configurations used for this test are: MLPRegressor(hidden_layer_sizes=(100,), learning_rate='adaptive', max_iter=500,solver='lbfgs')
if 7 in params["tests"]:
    exit_code = os.system(
        "python3 main.py estimator=\"ANN\" prediction_n=5 prediction_ex_search=True excelname=arria10_features.xlsx clustering_n=10 clustering_ex_search=False n_clusters=1 features=None target_metric=\"Clock Cycles\" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 clustering_min_features=1 prediction_min_features=2 prediction_max_features=None clustering_max_features=5 n_cpu=None parallel=True print=False regression_test=./test/test8.xlsx")
    total_exit_codes += exit_code
    total_tests += 1

if total_exit_codes == 0:
    print("\nALL REGRESSION TESTS PASSED!! :)\n")
    exit(0)
elif total_exit_codes == total_tests:
    print("\nWARNING: all regression tests FAILED! :(\n")
    exit(1)
else:
    print("\nWARNING: some regression tests FAILED! :(\n%d out of %d total tests FAILED :(\n" % (
    total_exit_codes, total_tests))
    exit(2)
