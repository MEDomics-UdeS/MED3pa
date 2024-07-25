import sys
import os

from MED3pa.datasets import DatasetsManager
from MED3pa.models import BaseModelManager, ModelFactory
from MED3pa.med3pa import Med3paExperiment

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

# Define parameters for the experiment
ipc_params = {'n_estimators': 100}
ipc_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf':[1, 2, 4]

}
apc_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf':[1, 2, 4]
}
apc_params = {'max_depth': 3}

# Execute the MED3PA experiment
results = Med3paExperiment.run(
                                datasets_manager=datasets,
                                base_model_manager=base_model_manager,
                                uncertainty_metric="absolute_error",
                                ipc_type='RandomForestRegressor',
                                ipc_params=ipc_params,
                                apc_params=apc_params,
                                ipc_grid_params=ipc_grid,
                                apc_grid_params=apc_grid,
                                samples_ratio_min=0,
                                samples_ratio_max=10,
                                samples_ratio_step=5,
                                evaluate_models=True,
                                )

# Save the results to a specified directory
results.save(file_path='./tests/tests_med3pa/med3pa_experiment_results')