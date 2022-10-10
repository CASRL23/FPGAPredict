#dedicated logic registers with 2 clusters 
python3 main.py estimator="Linear" excelname="arria10_features.xlsx" prediction_n=22 prediction_ex_search=True clustering_n=50 clustering_ex_search=1 n_clusters=2 features=None target_metric="Dedicated Logic Registers" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 prediction_min_features=2 prediction_max_features=None clustering_min_features=1 clustering_max_features=5 n_cpu=None parallel=True print=True regression_test=None >arria10_reg_2c_lr.txt

# run dedicated logic registers with 3 clusters
python3 main.py estimator="Linear" excelname="arria10_features.xlsx" prediction_n=22 prediction_ex_search=True clustering_n=50 clustering_ex_search=1 n_clusters=3 features=None target_metric="Dedicated Logic Registers" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 prediction_min_features=2 prediction_max_features=None clustering_min_features=1 clustering_max_features=5 n_cpu=None parallel=True print=True regression_test=None >arria10_reg_3c_lr.txt

# run Combinational ALUTs with 2 clusters
python3 main.py estimator="Linear" excelname="arria10_features.xlsx" prediction_n=22 prediction_ex_search=True clustering_n=50 clustering_ex_search=1 n_clusters=2 features=None target_metric="Combinational ALUTs" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 prediction_min_features=2 prediction_max_features=None clustering_min_features=1 clustering_max_features=5 n_cpu=None parallel=True print=True regression_test=None >arria10_aluts_2c_lr.txt
# run Combinational ALUTs with 3 clusters
python3 main.py estimator="Linear" excelname="arria10_features.xlsx" prediction_n=22 prediction_ex_search=True clustering_n=50 clustering_ex_search=1 n_clusters=3 features=None target_metric="Combinational ALUTs" clustering_features=None n_splits=5 random_state=0 n_runs=50 scaling=1 prediction_min_features=2 prediction_max_features=None clustering_min_features=1 clustering_max_features=5 n_cpu=None parallel=True print=True regression_test=None >arria10_aluts_3c_lr.txt

