from MED3pa.detectron.experiment import DetectronExperiment
from MED3pa.datasets.manager import DatasetsManager
from MED3pa.models.base import BaseModelManager
from MED3pa.models.factories import ModelFactory
from MED3pa.detectron.strategies import EnhancedDisagreementStrategy
import pandas as pd
import numpy as np
import json

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'nthread': 4,
    'tree_method': 'hist',
    'device': 'cpu'
}


# prepare the BaseModelManager
loaded_model = ModelFactory.create_model_from_pickled("./tests/tests_detectron/model.pkl")
bm_manager = BaseModelManager()
bm_manager.set_base_model(loaded_model)

# prepare the DatasetManager
datasets = DatasetsManager()
datasets.set_from_file("training","./tests/tests_detectron/cleveland_train.csv", "y_true")
datasets.set_from_file("validation", "./tests/tests_detectron/cleveland_val.csv", "y_true")
datasets.set_from_file("reference","./tests/tests_detectron/cleveland_test.csv", "y_true")
datasets.set_from_file("testing","./tests/tests_detectron/ood_va_sampled_seed_3.csv", "y_true")


detectron_results = DetectronExperiment.run(
    datasets=datasets, 
    training_params=XGB_PARAMS, 
    base_model_manager=bm_manager, 
)

analysis_results = detectron_results.analyze_results(EnhancedDisagreementStrategy)
detectron_results.save("./tests/tests_detectron/", "detectron_results")

