# run with the clock period, 2 clusters, and linear regression. Permutation with 10 features only.
python3 main.py estimator=1 n=10 perm=True excelname=arria10_features.xlsx clustering_n=10 clustering_perm=True n_clusters=2 features=None target_metric="Clock Period" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 clusters_features=None clustering_estimators=None clustering_perm_min=1 perm_min=2 perm_max=None clustering_perm_max n_cpu=None parallel=True print=False regression_test=./test/test1.xlsx

# run with the clock period, 2 clusters, and exponential regression. Permutation with 10 features only.
python3 main.py estimator=2 n=10 perm=True excelname=arria10_features.xlsx clustering_n=10 clustering_perm=True n_clusters=2 features=None target_metric="Clock Period" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 clusters_features=None clustering_estimators=None clustering_perm_min=1 perm_min=2 perm_max=None clustering_perm_max n_cpu=None parallel=True print=False regression_test=./test/test2.xlsx

# run with the clock period, 2 clusters, and random forest. Permutation with 10 features only.
python3 main.py estimator=3 n=10 perm=True excelname=arria10_features.xlsx clustering_n=10 clustering_perm=True n_clusters=2 features=None target_metric="Clock Period" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 clusters_features=None clustering_estimators=None clustering_perm_min=1 perm_min=2 perm_max=None clustering_perm_max n_cpu=None parallel=True print=False regression_test=./test/test3.xlsx
