from MED3pa.med3pa.experiment import Med3paExperiment
from MED3pa.datasets import DatasetsManager
import pandas as pd
import numpy as np

# Load data from CSV file
data = pd.read_csv('./tests/tests_med3pa/simulated_data.csv')
# Set MED3pa params
predicted_prob = data['pred_prob'].values
datasets_manager = DatasetsManager()
datasets_manager.set_from_file('reference','./tests/tests_med3pa/simulated_data_without_pred_prob.csv', 'y_true')
metrics = ['Auc', 'BalancedAccuracy', 'Auprc', 'F1Score']
max_depth_log = int(np.log2(datasets_manager.get_dataset_by_type('reference', True).get_observations().shape[0]))
ipc_param_grid = {
    'max_depth': range(2, max_depth_log + 1)
}
apc_param_grid = {
    'max_depth': 3
}
ipc_cv = min(4, int(datasets_manager.get_dataset_by_type('reference', True).get_observations().shape[0] / 2))
ref_results, test_results = Med3paExperiment.run(datasets_manager=datasets_manager, predicted_probabilities=predicted_prob, base_model_manager=None, ipc_grid_params=ipc_param_grid,
                                  ipc_cv=ipc_cv, apc_params=apc_param_grid, evaluate_models=False, samples_ratio_min = 0, samples_ratio_max=10, samples_ratio_step=5)

ref_results.save('./tests/tests_med3pa/med3pa_results')