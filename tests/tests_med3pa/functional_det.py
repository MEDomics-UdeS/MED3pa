from det3pa.models.factories import ModelFactory
from det3pa.models.base import BaseModelManager
from det3pa.datasets import DatasetsManager
from det3pa.med3pa.experiment import *

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
loaded_model = ModelFactory.create_model_from_pickled("./tests/tests_med3pa/model.pkl")

datasets = DatasetsManager()
datasets.set_base_model_training_data("./tests/tests_med3pa/bm_train_imputed.csv", "y_true")
datasets.set_base_model_validation_data("./tests/tests_med3pa/bm_validation_imputed.csv", "y_true")
datasets.set_reference_data("./tests/tests_med3pa/bm_test_imputed.csv", "y_true")
datasets.set_testing_data("./tests/tests_med3pa/ood_imputed.csv", "y_true")

x, y_true = datasets.get_reference_data()
x_ood, y_true_ood = datasets.get_testing_data()

features = datasets.column_labels
metrics = ['Auc', 'BalancedAccuracy', 'Auprc', 'F1Score']
bm_manager = BaseModelManager()
bm_manager.set_base_model(loaded_model)
results = Med3paDetectron.run(datasets=datasets,
                training_params= XGB_PARAMS,
                base_model_manager=bm_manager,
                uncertainty_metric=AbsoluteError,
                samples_ratio_min=-5, samples_ratio_step=5, samples_ratio_max=30,
                med3pa_metrics=metrics,
                samples_size= 20,
                ensemble_size=10,
                num_calibration_runs=100,
                patience=3,
                significance_level=0.05, 
                test_strategy=DisagreementStrategy_z_mean,
                allow_margin= False, 
                margin = 0.05,
                all_dr = False
                )

results.save('./tests/tests_med3pa/detectron_med3pa_results')
