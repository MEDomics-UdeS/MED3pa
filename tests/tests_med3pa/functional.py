from det3pa.med3pa.experiment import Med3paExperiment
import pandas as pd
import numpy as np

# Load data from CSV file
data = pd.read_csv('./tests/tests_med3pa/simulated_data.csv')

# Extract feature matrix and labels
X_samples = data[['x1', 'x2']].values
Y_true = data['y_true'].values
predicted_prob = data['pred_prob'].values
features = ['x1', 'x2']
metrics = ['Auc', 'BalancedAccuracy', 'Auprc', 'F1Score']
max_depth_log = int(np.log2(X_samples.shape[0]))
ipc_param_grid = {
    'max_depth': range(2, max_depth_log + 1)
}
apc_param_grid = {
    'max_depth': 3
}
ipc_cv = min(4, int(X_samples.shape[0] / 2))
_, results = Med3paExperiment.run(x=X_samples, y_true=Y_true, features=features, predicted_probabilities=predicted_prob, base_model_manager=None, ipc_grid_params=ipc_param_grid,
                                  ipc_cv=ipc_cv, apc_params=apc_param_grid, evaluate_models=True)

results.save('./tests/tests_med3pa/med3pa_results')