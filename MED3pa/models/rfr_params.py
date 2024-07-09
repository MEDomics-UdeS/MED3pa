rfr_params = [
    {"name": "n_estimators", "type": "int", "default": 100, "description": "The number of trees in the forest."},
    {"name": "criterion", "type": "string", "choices": ["squared_error", "absolute_error", "poisson"], "default": "squared_error", "description": "The function to measure the quality of a split."},
    {"name": "max_depth", "type": "int", "default": 4, "description": "The maximum depth of the tree. Increasing this value will make the model more complex."},
    {"name": "min_samples_split", "type": "int", "default": 2, "description": "The minimum number of samples required to split an internal node."},
    {"name": "min_samples_leaf", "type": "int", "default": 1, "description": "The minimum number of samples required to be at a leaf node."},
    {"name": "min_weight_fraction_leaf", "type": "float", "default": 0.0, "description": "The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node."},
    {"name": "max_features", "type": "string", "choices": ["auto", "sqrt", "log2"], "default": "auto", "description": "The number of features to consider when looking for the best split."},
    {"name": "max_leaf_nodes", "type": "int", "default": 100, "description": "Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity."},
    {"name": "min_impurity_decrease", "type": "float", "default": 0.0, "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value."},
    {"name": "bootstrap", "type": "bool", "default": True, "description": "Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree."},
    {"name": "oob_score", "type": "bool", "default": False, "description": "Whether to use out-of-bag samples to estimate the generalization score."},
    {"name": "n_jobs", "type": "int", "default": 1, "description": "The number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context."},
    {"name": "random_state", "type": "int", "default": 42, "description": "Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)."},
    {"name": "verbose", "type": "int", "default": 0, "description": "Controls the verbosity when fitting and predicting."},
    {"name": "warm_start", "type": "bool", "default": False, "description": "When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest."},
    {"name": "ccp_alpha", "type": "float", "default": 0.0, "description": "Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen."},
    {"name": "max_samples", "type": "float", "default": 0.9, "description": "If bootstrap is True, the number of samples to draw from X to train each base estimator."}
]

rfr_gridsearch_params = [
    {"name": "n_estimators", "type": "int", "default": [100, 200, 300, 400, 500], "description": "The number of trees in the forest."},  
    {"name": "max_depth", "type": "int", "default": [2, 3, 4, 5, 6], "description": "The maximum depth of the tree. Increasing this value will make the model more complex."},  
    {"name": "min_samples_split", "type": "int", "default": [2, 5, 10], "description": "The minimum number of samples required to split an internal node."},  
    {"name": "min_samples_leaf", "type": "int", "default": [1, 2, 4], "description": "The minimum number of samples required to be at a leaf node."},  
    {"name": "max_features", "type": "string", "default": ["auto", "sqrt", "log2"], "description": "The number of features to consider when looking for the best split."},  
    {"name": "bootstrap", "type": "bool", "default": [True, False], "description": "Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree."}  
]
