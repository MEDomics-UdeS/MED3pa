rfr_params = [
    {"name": "n_estimators", "type": "int", "default": 100},
    {"name": "criterion", "type": "string", "choices": ["squared_error", "absolute_error", "poisson"], "default": "squared_error"},
    {"name": "max_depth", "type": "int", "default": None},
    {"name": "min_samples_split", "type": "int", "default": 2},
    {"name": "min_samples_leaf", "type": "int", "default": 1},
    {"name": "min_weight_fraction_leaf", "type": "float", "default": 0.0},
    {"name": "max_features", "type": "string", "choices": ["auto", "sqrt", "log2"], "default": "auto"},
    {"name": "max_leaf_nodes", "type": "int", "default": None},
    {"name": "min_impurity_decrease", "type": "float", "default": 0.0},
    {"name": "bootstrap", "type": "bool", "default": True},
    {"name": "oob_score", "type": "bool", "default": False},
    {"name": "n_jobs", "type": "int", "default": None},
    {"name": "random_state", "type": "int", "default": None},
    {"name": "verbose", "type": "int", "default": 0},
    {"name": "warm_start", "type": "bool", "default": False},
    {"name": "ccp_alpha", "type": "float", "default": 0.0},
    {"name": "max_samples", "type": "float", "default": None}
]

rfr_gridsearch_params = [
    {"name": "n_estimators", "type": "int", "default": [100, 200, 300, 400, 500]},  
    {"name": "max_depth", "type": "int", "default": [None, 10, 20, 30, 40, 50]},  
    {"name": "min_samples_split", "type": "int", "default": [2, 5, 10]},  
    {"name": "min_samples_leaf", "type": "int", "default": [1, 2, 4]},  
    {"name": "max_features", "type": "string", "default": ["auto", "sqrt", "log2"]},  
    {"name": "bootstrap", "type": "bool", "default": [True, False]}  
]