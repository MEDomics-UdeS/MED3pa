from MED3pa.models.factories import ModelFactory
from MED3pa.models.base import BaseModelManager
from MED3pa.datasets import DatasetsManager
from MED3pa.med3pa.experiment import *

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
datasets.set_from_file(dataset_type="training", file ="./tests/tests_med3pa/bm_train_imputed.csv", target_column_name="y_true")
datasets.set_from_file("validation","./tests/tests_med3pa/bm_validation_imputed.csv", "y_true")
datasets.set_from_file("reference","./tests/tests_med3pa/bm_test_imputed.csv", "y_true")
datasets.set_from_file("testing","./tests/tests_med3pa/ood_imputed.csv", "y_true")

x, y_true = datasets.get_dataset_by_type("reference")
x_ood, y_true_ood = datasets.get_dataset_by_type("testing")

features = datasets.column_labels
metrics = ['Auc', 'BalancedAccuracy', 'Auprc', 'F1Score']
bm_manager = BaseModelManager()
bm_manager.set_base_model(loaded_model)
reference_3pa_res, testing_3pa_res, detectron_res = Med3paDetectronExperiment.run(datasets=datasets,
                                                                        training_params= XGB_PARAMS,
                                                                        base_model_manager=bm_manager,
                                                                        uncertainty_metric=AbsoluteError,
                                                                        samples_ratio_min=-5, samples_ratio_step=5, samples_ratio_max=30,
                                                                        med3pa_metrics=metrics,
                                                                        samples_size= 20,
                                                                        ensemble_size=10,
                                                                        num_calibration_runs=100,
                                                                        patience=3,
                                                                        test_strategy=EnhancedDisagreementStrategy,
                                                                        allow_margin= False, 
                                                                        margin = 0.05,
                                                                        all_dr = False
                                                                        )

reference_3pa_res.save('./tests/tests_med3pa/detectron_med3pa_results/reference_3pa_results')
testing_3pa_res.save('./tests/tests_med3pa/detectron_med3pa_results/testing_3pa_results')
detectron_res.save('./tests/tests_med3pa/detectron_med3pa_results')