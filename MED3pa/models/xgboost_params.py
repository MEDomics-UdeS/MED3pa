valid_xgboost_params = {
    "booster",
    "nthread",
    "verbosity",
    "validate_parameters",
    "device",
    "seed",
    "seed_per_iteration",
    "disable_default_eval_metric",

    # Tree Booster parameters
    "eta",
    "gamma",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "colsample_bylevel",
    "colsample_bynode",
    "lambda",
    "alpha",
    "tree_method",
    "scale_pos_weight",
    "max_bin",
    "sampling_method",
    "grow_policy",
    "max_leaves",
    "max_delta_step",
    "refresh_leaf",
    "process_type",
    "updater",
    "monotone_constraints",
    "interaction_constraints",

    # DART Booster parameters
    "sample_type",
    "normalize_type",
    "rate_drop",
    "one_drop",
    "skip_drop",

    # Linear Booster parameters
    "feature_selector",
    "top_k",

    # Learning task parameters
    "objective",
    "base_score",
    "eval_metric",

    # Special objective parameters
    "tweedie_variance_power",
    "huber_slope",
    "quantile_alpha",
    "aft_loss_distribution",

    # Rank parameters
    "lambdarank_pair_method",
    "lambdarank_num_pair_per_sample",
    "lambdarank_unbiased",
    "lambdarank_bias_norm",
    "ndcg_exp_gain", 
}

valid_xgboost_custom_params = {
    # custom parameters
    "custom_eval_metrics", 
    "num_boost_rounds",
    "training_weights"
}

xgboost_params = [
    {"name": "booster", "type": "string", "choices": ["gbtree", "gblinear", "dart"], "default": "gbtree"},
    {"name": "nthread", "type": "int", "default": None},
    {"name": "verbosity", "type": "int", "choices": [0, 1, 2, 3], "default": 1},
    {"name": "validate_parameters", "type": "bool", "default": False},
    {"name": "device", "type": "string", "choices": ["cpu", "cuda"], "default": "cpu"},
    {"name": "seed", "type": "int", "default": 0},
    {"name": "seed_per_iteration", "type": "bool", "default": 0},
    {"name": "disable_default_eval_metric", "type": "bool", "default": 0},

    # Tree Booster parameters
    {"name": "eta", "type": "float", "range": [0, 1], "default": 0.3},
    {"name": "gamma", "type": "float", "range": [0, float('inf')], "default": 0},
    {"name": "max_depth", "type": "int", "range": [0, float('inf')], "default": 6},
    {"name": "min_child_weight", "type": "float", "range": [0, float('inf')], "default": 1},
    {"name": "subsample", "type": "float", "range": [0, 1], "default": 1},
    {"name": "colsample_bytree", "type": "float", "range": [0, 1], "default": 1},
    {"name": "colsample_bylevel", "type": "float", "range": [0, 1], "default": 1},
    {"name": "colsample_bynode", "type": "float", "range": [0, 1], "default": 1},
    {"name": "lambda", "type": "float", "range": [0, float('inf')], "default": 1},
    {"name": "alpha", "type": "float", "range": [0, float('inf')], "default": 0},
    {"name": "tree_method", "type": "string", "choices": ["auto", "exact", "approx", "hist", "gpu_hist"], "default": "auto"},
    {"name": "scale_pos_weight", "type": "float", "default": 1},
    {"name": "max_bin", "type": "int", "default": 256},
    {"name": "sampling_method", "type": "string", "choices": ["uniform", "gradient_based"], "default": "uniform"},
    {"name": "grow_policy", "type": "string", "choices": ["depthwise", "lossguide"], "default": "depthwise"},
    {"name": "max_leaves", "type": "int", "default": 0},
    {"name": "max_delta_step", "type": "float", "default": 0},
    {"name": "refresh_leaf", "type": "int", "choices": [0, 1], "default": 1},
    {"name": "process_type", "type": "string", "choices": ["default", "update"], "default": "default"},
    {"name": "updater", "type": "string", "default": "grow_colmaker,prune"},
    {"name": "monotone_constraints", "type": "string", "default": "()"},
    {"name": "interaction_constraints", "type": "string", "default": ""},

    # DART Booster parameters
    {"name": "sample_type", "type": "string", "choices": ["uniform", "weighted"], "default": "uniform"},
    {"name": "normalize_type", "type": "string", "choices": ["tree", "forest"], "default": "tree"},
    {"name": "rate_drop", "type": "float", "range": [0, 1], "default": 0.0},
    {"name": "one_drop", "type": "bool", "default": 0},
    {"name": "skip_drop", "type": "float", "range": [0, 1], "default": 0.0},

    # Linear Booster parameters
    {"name": "updater", "type": "string", "choices": ["shotgun", "coord_descent"], "default": "shotgun"},
    {"name": "feature_selector", "type": "string", "choices": ["cyclic", "shuffle", "random", "greedy", "thrifty"], "default": "cyclic"},
    {"name": "top_k", "type": "int", "default": 0},

    # Learning task parameters
    {"name": "objective", "type": "string", "choices": [
        "reg:squarederror", "reg:squaredlogerror", "reg:logistic", "reg:pseudohubererror",
        "reg:absoluteerror", "reg:quantileerror", "binary:logistic", "binary:logitraw",
        "binary:hinge", "count:poisson", "survival:cox", "survival:aft", "multi:softmax",
        "multi:softprob", "rank:ndcg", "rank:map", "rank:pairwise", "reg:gamma", "reg:tweedie"
    ], "default": "reg:squarederror"},
    {"name": "base_score", "type": "float", "default": 0.5},
    {"name": "eval_metric", "type": "string", "default": "rmse"},

    # Special objective parameters
    {"name": "tweedie_variance_power", "type": "float", "range": [1, 2], "default": 1.5},
    {"name": "huber_slope", "type": "float", "default": 1.0},
    {"name": "quantile_alpha", "type": "float", "default": 0.5},
    {"name": "aft_loss_distribution", "type": "string", "choices": ["normal", "logistic", "extreme"], "default": "normal"},

    # Rank parameters
    {"name": "lambdarank_pair_method", "type": "string", "choices": ["mean", "topk"], "default": "mean"},
    {"name": "lambdarank_num_pair_per_sample", "type": "int", "default": 1},
    {"name": "lambdarank_unbiased", "type": "bool", "default": False},
    {"name": "lambdarank_bias_norm", "type": "float", "default": 2.0},
    {"name": "ndcg_exp_gain", "type": "bool", "default": True}
]

# Translation mapping for xgboost metric names to implemented metrics
xgboost_metrics = {
    'auc': 'Auc',
    'logloss': 'LogLoss',
    'aucpr': 'Auprc',
}
