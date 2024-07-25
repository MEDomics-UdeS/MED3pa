from MED3pa.detectron.experiment import DetectronExperiment
from MED3pa.datasets.manager import DatasetsManager
from MED3pa.models.base import BaseModelManager
from MED3pa.models.factories import ModelFactory
from MED3pa.detectron.strategies import EnhancedDisagreementStrategy
from MED3pa.datasets import DatasetsManager

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

# Initialize the DatasetsManager
datasets = DatasetsManager()

# Load datasets for reference, and testing
datasets.set_from_file(dataset_type="training", file='./tests/tests_med3pa/data/train_data.csv', target_column_name='Outcome')
datasets.set_from_file(dataset_type="validation", file='./tests/tests_med3pa/data/val_data.csv', target_column_name='Outcome')
datasets.set_from_file(dataset_type="reference", file='./tests/tests_med3pa/data/test_data.csv', target_column_name='Outcome')
datasets.set_from_file(dataset_type="testing", file='./tests/tests_med3pa/data/test_data_shifted_0.1.csv', target_column_name='Outcome')

# Initialize the model factory and load the pre-trained model
factory = ModelFactory()
model = factory.create_model_from_pickled("./tests/tests_med3pa/models/diabetes_xgb_model.pkl")

# Set the base model using BaseModelManager
base_model_manager = BaseModelManager()
base_model_manager.set_base_model(model=model)

experiment_results = DetectronExperiment.run(datasets=datasets, base_model_manager=base_model_manager, training_params=XGB_PARAMS)

analysis_results = experiment_results.analyze_results("enhanced_disagreement_strategy")
experiment_results.save("./tests/tests_detectron/detectron_results")

