# FPGAPredict Research
Given a set of benchmarks with their CPU and FPGA features, FPGAPredict can perform clustering, feature selection, and KFold cross validation to predict a specific target metric on the FPGA (i.e. execution time or total power). It also can perform multiple random runs verification with KFold and prints the average error of these runs to verify that the results are accurate even with random KFolds.

## FPGAPredict Usage

To edit configurations, either edit config.json or pass parameters as inputs to command line.
Passed paramaters through command line will override any configurations in config.json.

### Parameters Definitions

*estimator*: (default: 1) an integer from the following dictionary:
 		    {1: 'LinearRegression', 2: 'ExponentialRegression', 3: 'RandomForestRegressor', 4: 'MLPRegressor', 5: 'DecisionTreeRegressor', 6: 'KNeighborsRegressor'}

*excelname*: (default: 'arria2_features.xls') the name of the excel file that contains the data, must have an extension of xls or xlsx.

*n*: (default: 22) the number of top PCC ranked prediction features to pass to permutation, only used when perm=True.

*perm* (default: False): boolean indicates whether to select prediction features by permutation or not.

*clustering_n* (default: 50): the number of top PCC ranked CLUSTERING features to pass to permutation. Only used when clustering_perm=True.

*clustering_perm* (default: False): boolean indicates whether to select CLUSTERING features by permutation or not.

*n_clusters* (default: 1): the number of clusters to create. If 1 then clustering is not performed.

*features* (default: None i.e. all features): An array of strings where each string indicates the features to use for prediction. If None (default), all features are passed.

*target_feature* (default: "Clock Cycles"): A string of the target feature to predict.

*clustering_features* (default: None i.e. all features): An array of strings where each string indicates the features to use for CLUSTERING. If None (default), all features are passed.

*n_splits* (default: 5): Number of KFold n_splits. If n_splits <= 1 then KFold isn't performed and the results will show the training error only.
*shuffle* (default: True): whether to perform true shuffle or seeded shuffle with KFold. If True then true shuffle is performed, but if False then seeded shuffle is performed. Not used if n_splits <= 1.

*random_state* (default: None): The starting seed of the shuffle. Only used if shuffle=False. This variable is incremented with every run performed (i.e. it is incremented <n_runs> times).

*perm_up_to* (default: None): an integer or None, the maximum number of features to try with prediction exhaustive search. If None then this number is equal to n. Only used when perm=True.

*clustering_perm_up_to* (default: 5): an integer or None, the maximum number of clustering features to try in permutation. If None then this number is equal to clustering_n. Only used when clustering_perm=True.

*perm_min* (default: 1): an integer, the minimum number of features to try with prediction exhaustive search. 

*clustering_perm_min* (default: 1): an integer, the minimum number of clustering features to try with clustering exhaustive search. 

*n_runs* (default: 50): an integer, the number of random runs to perform to record their results. Every run is performed with a different seed to make sure each run leads to a different random split.

*n_cpu* (default: None): an integer or None, the number of parallel CPU runs to perform. If None, the code calculates the number of available cores and subtracts 3. Only used if parallel=True and either perm or clustering_perm are True.

*parallel* (default: True): boolean, whether to perform parallel processing with exhaustive search or not. Only used if either perm=True or clustering_perm=True.

*print* (default: True): boolean, whether to print details of the runs and their results. Regardless of this variable, the main will store all results in excel files even if they are not printed (i.e. print=False).

### Results
Four main outputs are stored:
1)  A directory of the target feature will be created (if not there already). 
    Under this directory, an excel file will be stored of the format: <estimator>_<board>_c<n_clusters>.xlsx
    This file will have one random run of the clusters, each of them separately (without the clustering prediction part)
    And it will have another sheet of <n_runs> (i.e. 50) random runs that include the clustering prediction. 
    These runs will record the average error for each single benchmark and a summary of
    the average <n_runs> error, STDEV, RMSE, and RRMSE. Also it will record the stdev <n_runs> of the error, STDEV, RMSE, and RRMSE.
2)  In the main directory of the project, the same file in number 1 will be stored. 
    However, this file may be overwritten if the same experiment is performed, while the file in number 1 will never be overwritten.
3)  If prediction exhaustive search was performed, then in the directory of the target feature, a file will be stored that contain the top 5 prediction features subsets.
    The file has a name format of "perm_results <Target Feature>_<clustering or prediction>_<estimator>.csv"
4)  If clustering exhaustive search was performed, then in the directory of the target feature, a file will be stored that contain all clustering features subset that gave a clustering error of 0%. If none of them gave 0% clustering error, the clustering features with the lowest error is stored only.

### Examples

Use the following command to collect both clustering features with exhaustive search and prediction features with exhaustive search as well for a specific board and with a specific estimator. 

	python3 main.py excelname=<string> estimator=1 n=22 perm=True clustering_n=22 clustering_perm=True n_clusters=2 target_feature=<string> shuffle=True perm_shuffle=True random_state=None &><filename> &
	
		
For example, 
	
	python3 main.py excelname=arria10_features.xls target_feature='Clock Period' shuffle=True random_state=None perm_shuffle=True clustering_perm=True clustering_n=22 n_clusters=2 perm=True n=22 scaling=1 estimator=1

This command will find the clustering features and prediction features to predict the clock period on Arria 10 with 2 clusters and with linear regression.
