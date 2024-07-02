dtr_params = [
    {"name": "criterion", "type": "string", "default": "squared_error", "choices": ["squared_error", "friedman_mse", "absolute_error", "poisson"]},
    {"name": "splitter", "type": "string", "default": "best", "choices": ["best", "random"]},
    {"name": "max_depth", "type": "int", "default": None},
    {"name": "min_samples_split", "type": "int", "default": 2},
    {"name": "min_samples_leaf", "type": "int", "default": 1},
    {"name": "min_weight_fraction_leaf", "type": "float", "default": 0.0},
    {"name": "max_features", "type": "string", "default": None, "choices": [None, "auto", "sqrt", "log2"]},
    {"name": "max_leaf_nodes", "type": "int", "default": None},
    {"name": "min_impurity_decrease", "type": "float", "default": 0.0},
    {"name": "ccp_alpha", "type": "float", "default": 0.0}
]

dtr_gridsearch_params = [
    {"name": "criterion", "type": "string", "default": ["squared_error", "friedman_mse", "absolute_error", "poisson"]},  
    {"name": "splitter", "type": "string", "default": ["best", "random"]},  
    {"name": "max_depth", "type": "int", "default": [None, 10, 20, 30, 40, 50]},  
    {"name": "min_samples_split", "type": "int", "default": [2, 5, 10]},  
    {"name": "min_samples_leaf", "type": "int", "default": [1, 2, 4]},  
    {"name": "min_weight_fraction_leaf", "type": "float", "default": [0.0, 0.1, 0.2]},  
    {"name": "max_features", "type": "string", "default": [None, "auto", "sqrt", "log2"]},  
    {"name": "max_leaf_nodes", "type": "int", "default": [None, 10, 20, 30]},  
    {"name": "min_impurity_decrease", "type": "float", "default": [0.0, 0.1, 0.2]}, 
    {"name": "ccp_alpha", "type": "float", "default": [0.0, 0.01, 0.1]}  
]