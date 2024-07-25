import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from MED3pa.datasets import DatasetsManager
from MED3pa.models import BaseModelManager, ModelFactory
from MED3pa.med3pa import Med3paDetectronExperiment

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
med3pa_metrics = ['Auc', 'Accuracy', 'BalancedAccuracy']

# Execute the MED3PA experiment
# Execute the integrated MED3PA and Detectron experiment
med3pa_detectron_results = Med3paDetectronExperiment.run(
                                datasets=datasets,
                                base_model_manager=base_model_manager,
                                uncertainty_metric="absolute_error",
                                samples_size=20,
                                ensemble_size=10,
                                num_calibration_runs=100,
                                patience=3,
                                test_strategies="enhanced_disagreement_strategy",
                                allow_margin=False,
                                margin=0.05,
                                samples_ratio_min=0,
                                samples_ratio_max=10,
                                samples_ratio_step=5,
                                evaluate_models=True,
)

# Save the results to a specified directory
med3pa_detectron_results.save(file_path='./tests/tests_med3pa/med3pa_detectron_experiment_results')