dtr_params = [
    {"name": "criterion", "type": "string", "default": "squared_error", "choices": ["squared_error", "friedman_mse", "absolute_error", "poisson"], "description": "The function to measure the quality of a split."},
    {"name": "splitter", "type": "string", "default": "best", "choices": ["best", "random"], "description": "The strategy used to choose the split at each node."},
    {"name": "max_depth", "type": "int", "default": 4, "description": "The maximum depth of the tree. Increasing this value will make the model more complex."},
    {"name": "min_samples_split", "type": "int", "default": 2, "description": "The minimum number of samples required to split an internal node."},
    {"name": "min_samples_leaf", "type": "int", "default": 1, "description": "The minimum number of samples required to be at a leaf node."},
    {"name": "min_weight_fraction_leaf", "type": "float", "default": 0.0, "description": "The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node."},
    {"name": "max_features", "type": "string", "default": "auto", "choices": ["auto", "sqrt", "log2"], "description": "The number of features to consider when looking for the best split."},
    {"name": "max_leaf_nodes", "type": "int", "default": 100, "description": "Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity."},
    {"name": "min_impurity_decrease", "type": "float", "default": 0.0, "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value."},
    {"name": "ccp_alpha", "type": "float", "default": 0.0, "description": "Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen."}
]

dtr_gridsearch_params = [
    {"name": "criterion", "type": "string", "default": ["squared_error", "friedman_mse", "absolute_error", "poisson"], "description": "The function to measure the quality of a split."},
    {"name": "splitter", "type": "string", "default": ["best", "random"], "description": "The strategy used to choose the split at each node."},
    {"name": "max_depth", "type": "int", "default": [2, 3, 4, 5, 6], "description": "The maximum depth of the tree. Increasing this value will make the model more complex."},
    {"name": "min_samples_split", "type": "int", "default": [2, 5, 10], "description": "The minimum number of samples required to split an internal node."},
    {"name": "min_samples_leaf", "type": "int", "default": [1, 2, 4], "description": "The minimum number of samples required to be at a leaf node."},
    {"name": "min_weight_fraction_leaf", "type": "float", "default": [0.0, 0.1, 0.2], "description": "The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node."},
    {"name": "max_features", "type": "string", "default": ["auto", "sqrt", "log2"], "description": "The number of features to consider when looking for the best split."},
    {"name": "max_leaf_nodes", "type": "int", "default": [10, 20, 30], "description": "Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity."},
    {"name": "min_impurity_decrease", "type": "float", "default": [0.0, 0.1, 0.2], "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value."},
    {"name": "ccp_alpha", "type": "float", "default": [0.0, 0.01, 0.1], "description": "Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen."}
]
