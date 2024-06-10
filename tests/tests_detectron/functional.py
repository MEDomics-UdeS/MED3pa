from det3pa.detectron.experiment import DetectronExperiment
from det3pa.datasets.manager import DatasetsManager
from det3pa.models.base import BaseModelManager
from det3pa.models.factories import ModelFactory
from det3pa.detectron.strategies import DisagreementStrategy, DisagreementStrategy_MW, DisagreementStrategy_quantile, DisagreementStrategy_z_mean
import pandas as pd
import numpy as np
import json


def evaluate_model_on_splits(model, split_files, metrics):
    for split_file in split_files:
        # Load the CSV file
        df = pd.read_csv(split_file)
        
        # Extract features and target
        x_test = df.drop(columns=['y_true']).values
        y_test = df['y_true'].values
        
        # Evaluate the model
        print(f"Evaluating on {split_file}")
        model.evaluate(x_test, y_test, metrics, True)


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

loaded_model = ModelFactory.create_model_from_pickled("./tests/tests_detectron/model.pkl")

datasets = DatasetsManager()
datasets.set_base_model_training_data("./tests/tests_detectron/cleveland_train.csv", "y_true")
datasets.set_base_model_validation_data("./tests/tests_detectron/cleveland_val.csv", "y_true")
datasets.set_reference_data("./tests/tests_detectron/cleveland_test.csv", "y_true")

x, y = datasets.get_reference_data()

bm_manager = BaseModelManager()
bm_manager.set_base_model(loaded_model)

experiment = DetectronExperiment
print("///////////////////////// Base model evaluation on the different datasets : ////////////////////////////////////////")

# List of CSV files
split_files = [f'./tests/tests_detectron/ood_va_sampled_seed_{seed}.csv' for seed in range(10)]

# Only evaluate on splits 1 and 4
split_files_to_evaluate = [split_files[0], split_files[3], split_files[7]]

evaluate_model_on_splits(loaded_model, split_files=split_files_to_evaluate, metrics=['Auc', 'BalancedAccuracy'])

print("///////////////////////// Detectron experiment with different tests on unshifted splits: ////////////////////////////////////////")

# Initialize a list to store results for each split
all_results = []

# Running experiments for splits 1 and 4 only
for i, split_file in enumerate(split_files_to_evaluate):
    print(f"Running Detectron experiments on {split_file}")
    # Load the CSV file
    df = pd.read_csv(split_file)
    
    # Set the split data as the testing data
    datasets.set_testing_data(split_file, "y_true")

    print("Running Detectron using DisagreementStrategy")
    detectron_results, exp_res, eval_res = DetectronExperiment.run(
        datasets=datasets, 
        calib_result=cal_rec if i > 0 else None, 
        training_params=XGB_PARAMS, 
        base_model_manager=bm_manager, 
        test_strategy=DisagreementStrategy
    )
    if i == 0:
        cal_rec = detectron_results.cal_record
        samples_counts = cal_rec.sampling_counts
       
    # Save the experimental results and rejection counts for this split
    split_result = {
        "split_file": split_file,
        "disagreement_exp_res": exp_res,
    }

    print("Running Detectron using DisagreementStrategy_Quantile")
    _, exp_res, eval_res = DetectronExperiment.run(
        datasets=datasets, 
        detectron_result=detectron_results, 
        training_params=XGB_PARAMS, 
        base_model_manager=bm_manager, 
        test_strategy=DisagreementStrategy_quantile
    )
    print(exp_res)
    
    # Save the results for DisagreementStrategy_KS
    split_result.update({
        "disagreement_quantile_exp_res": exp_res,
    })

    print("Running Detectron using DisagreementStrategy_MW")
    _, exp_res, eval_res = DetectronExperiment.run(
        datasets=datasets, 
        detectron_result=detectron_results, 
        training_params=XGB_PARAMS, 
        base_model_manager=bm_manager, 
        test_strategy=DisagreementStrategy_MW
    )
    
    split_result.update({
        "disagreement_mw_exp_res": exp_res,
    })
    print("Running Detectron using EnhancedDisagreementTest")
    _, exp_res, eval_res = DetectronExperiment.run(
        datasets=datasets, 
        detectron_result=detectron_results, 
        training_params=XGB_PARAMS, 
        base_model_manager=bm_manager, 
        test_strategy=DisagreementStrategy_z_mean
    )
    print(exp_res)
    
    # Save the results for EnhancedDisagreementTest
    split_result.update({
        "disagreement_enhanced_exp_res": exp_res,
        "cal_record_rejection_counts": detectron_results.cal_record.rejected_counts(),
        "test_record_rejection_counts": detectron_results.test_record.rejected_counts()
    })

    print(detectron_results.cal_record.rejected_counts())
    print(detectron_results.test_record.rejected_counts())

    # Append the split result to all_results
    all_results.append(split_result)

# Function to convert numpy objects to native Python objects
def convert_numpy_objects(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_objects(item) for item in obj]
    return obj

# Convert all_results to native Python objects
all_results = convert_numpy_objects(all_results)

# Save results to a JSON file
with open("./tests/tests_detectron/detectron_results.json", "w") as json_file:
    json.dump(all_results, json_file, indent=4)

