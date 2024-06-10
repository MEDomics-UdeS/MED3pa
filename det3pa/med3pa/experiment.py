"""
This module include classes and methods to run and store the results of med3pa and med3pa-detectron experiments.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from det3pa.models.classification_metrics import *
from det3pa.med3pa.tree import TreeRepresentation
from det3pa.med3pa.profiles import ProfilesManager, Profile
from det3pa.datasets.manager import DatasetsManager, MaskedDataset
from det3pa.models.base import BaseModelManager
from det3pa.models.concrete_regressors import *
from det3pa.med3pa.uncertainty import *
from det3pa.med3pa.models import *
from det3pa.med3pa.calculator import MDRCalculator
from det3pa.detectron.experiment import DetectronExperiment, DisagreementStrategy_z_mean, DetectronResult
import json
import os
from typing import Any, Dict, List, Tuple, Type


def to_serializable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable format.

    Args:
        obj (Any): The object to convert.

    Returns:
        Any: The JSON-serializable representation of the object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Profile):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


class Med3paResults:
    """
    Class to store and manage results from the MED3PA method.
    """
    def __init__(self) -> None:
        self.metrics_by_dr: Dict[int, Dict] = {}
        self.profiles_records: Dict[int, Dict] = {}
        self.lost_profiles_records: Dict[int, Dict] = {}
        self.models_evaluation: Dict[str, Dict] = {}

    def set_metrics_by_dr(self, samples_ratio: int, metrics_by_dr: Dict) -> None:
        """
        Set metrics by declaration rate for a given sample ratio.

        Args:
            samples_ratio (int): The sample ratio.
            metrics_by_dr (Dict): Dictionary of metrics by declaration rate.
        """
        if samples_ratio not in self.metrics_by_dr:
            self.metrics_by_dr[samples_ratio] = {}
        self.metrics_by_dr[samples_ratio] = metrics_by_dr
    
    def set_profiles(self, samples_ratio: int, profiles: Dict, lost_profiles: Dict) -> None:
        """
        Set profiles and lost profiles for a given sample ratio.

        Args:
            samples_ratio (int): The sample ratio.
            profiles (Dict): Dictionary of profiles.
            lost_profiles (Dict): Dictionary of lost profiles.
        """
        if samples_ratio not in self.profiles_records:
            self.profiles_records[samples_ratio] = profiles[samples_ratio]
            self.lost_profiles_records[samples_ratio] = lost_profiles[samples_ratio]

    def set_models_evaluation(self, ipc_evaluation: Dict, apc_evaluation: Dict) -> None:
        """
        Set models evaluation metrics.

        Args:
            ipc_evaluation (Dict): Evaluation metrics for IPC model.
            apc_evaluation (Dict): Evaluation metrics for APC model.
        """
        self.models_evaluation['IPC_evaluation'] = ipc_evaluation
        self.models_evaluation['APC_evaluation'] = apc_evaluation
    
    def save(self, file_path: str) -> None:
        """
        Save the metrics and profiles to JSON files.

        Args:
            file_path (str): The file path to save the JSON files.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)

        with open(f'{file_path}/metrics_dr.json', 'w') as file:
            json.dump(self.metrics_by_dr, file, default=to_serializable, indent=4)
        with open(f'{file_path}/profiles.json', 'w') as file:
            json.dump(self.profiles_records, file, default=to_serializable, indent=4)
        with open(f'{file_path}/lost_profiles.json', 'w') as file:
            json.dump(self.lost_profiles_records, file, default=to_serializable, indent=4)
        if self.models_evaluation is not None:
            with open(f'{file_path}/models_evaluation.json', 'w') as file:
                json.dump(self.models_evaluation, file, default=to_serializable, indent=4)


class Med3paExperiment:
    """
    Class to run the MED3PA method experiment.
    """
    @staticmethod
    def run(x: np.ndarray,
            y_true: np.ndarray,
            features: List[str],
            predicted_probabilities: np.ndarray = None,
            base_model_manager: BaseModelManager = None,
            uncertainty_metric: Type[UncertaintyMetric] = AbsoluteError,
            ipc_type: Type[RegressionModel] = RandomForestRegressorModel,
            ipc_params: Dict = None,
            ipc_grid_params: Dict = None, 
            ipc_cv: int = None,
            apc_params: Dict = None,
            apc_grid_params: Dict = None,
            apc_cv: int = None,
            samples_ratio_min: int = 0,
            samples_ratio_max: int = 50, 
            samples_ratio_step: int = 5,
            med3pa_metrics: List[str] = ['Auc', 'Accuracy', 'BalancedAccuracy'],
            evaluate_models: bool = False,
            models_metrics: List[str] = ['MSE', 'RMSE']) -> Tuple[np.ndarray, Med3paResults]:
        """Orchestrates the MED3PA experiment.

        Args:
            x (np.ndarray): Input features.
            y_true (np.ndarray): True labels.
            features (list of str): List of feature names.
            predicted_probabilities (np.ndarray, optional): Predicted probabilities, by default None.
            base_model_manager (BaseModelManager, optional): Instance of BaseModelManager to get the base model, by default None.
            uncertainty_metric (Type[UncertaintyMetric], optional): Instance of UncertaintyMetric to calculate uncertainty, by default AbsoluteError.
            ipc_type (Type[RegressionModel], optional): The class of the regressor model to use for IPC, by default RandomForestRegressorModel.
            ipc_params (dict, optional): Parameters for initializing the IPC regressor model, by default None.
            ipc_grid_params (dict, optional): Grid search parameters for optimizing the IPC model, by default None.
            ipc_cv (int, optional): Number of cross-validation folds for optimizing the IPC model, by default None.
            apc_params (dict, optional): Parameters for initializing the APC regressor model, by default None.
            apc_grid_params (dict, optional): Grid search parameters for optimizing the APC model, by default None.
            apc_cv (int, optional): Number of cross-validation folds for optimizing the APC model, by default None.
            samples_ratio_min (int, optional): Minimum sample ratio, by default 0.
            samples_ratio_max (int, optional): Maximum sample ratio, by default 50.
            samples_ratio_step (int, optional): Step size for sample ratio, by default 5.
            med3pa_metrics (list of str, optional): List of metrics to calculate, by default ['Auc', 'Accuracy', 'BalancedAccuracy'].
            evaluate_models (bool, optional): Whether to evaluate the models, by default False.
            models_metrics (list of str, optional): List of metrics for model evaluation, by default ['MSE', 'RMSE'].

        Returns:
            Tuple[np.ndarray, Med3paResults]: MPC values and the results of the MED3PA experiment.
        """
        # Initialize base model and predict probabilities if not provided
        if base_model_manager is None and predicted_probabilities is None:
            raise ValueError("Either the base model or the predicted probabilities should be provided!")
        
        if predicted_probabilities is None:
            base_model = base_model_manager.get_instance()
            predicted_probabilities = base_model.predict(x, True)

        # Calculate uncertainty values
        uncertainty_calc = UncertaintyCalculator(uncertainty_metric)
        uncertainty_values = uncertainty_calc.calculate_uncertainty(x, predicted_probabilities, y_true)

        # Predict binary labels
        y_pred = (predicted_probabilities >= 0.5).astype(int)

        if evaluate_models:
            x_train, x_test, uncertainty_train, uncertainty_test = train_test_split(x, uncertainty_values, test_size=0.1, random_state=42)
        else:
            x_train = x
            uncertainty_train = uncertainty_values

        # Create and train IPCModel
        IPC_model = IPCModel(ipc_type, ipc_params)
        IPC_model.train(x_train, uncertainty_train)
        if ipc_grid_params is not None:
            IPC_model.optimize(ipc_grid_params, ipc_cv, x_train, uncertainty_train)

        # Predict IPC values
        IPC_values = IPC_model.predict(x)

        # Create and train APCModel
        APC_model = APCModel(features, apc_params)
        APC_model.train(x, IPC_values)
        if apc_grid_params is not None:
            APC_model.optimize(apc_grid_params, apc_cv, x_train, uncertainty_train)

        results = Med3paResults()
        for samples_ratio in range(samples_ratio_min, samples_ratio_max + 1, samples_ratio_step):
            # Predict APC values
            APC_values = APC_model.predict(x, min_samples_ratio=samples_ratio)

            # Create and predict MPC values
            MPC_model = MPCModel(IPC_values, APC_values)
            MPC_values = MPC_model.predict()

            # Calculate metrics by declaration rate
            metrics_by_dr = MDRCalculator.calc_metrics_by_dr(
                y_true, y_pred, predicted_probabilities, MPC_values, metrics_list=med3pa_metrics)
            results.set_metrics_by_dr(samples_ratio, metrics_by_dr)

            # Calculate profiles and their metrics by declaration rate
            tree = APC_model.treeRepresentation
            profiles_manager = ProfilesManager(features)
            MDRCalculator.calc_profiles(profiles_manager, tree, MPC_values, samples_ratio)
            MDRCalculator.calc_metrics_by_profiles(profiles_manager, x, y_true, predicted_probabilities, y_pred, MPC_values, med3pa_metrics)

            results.set_profiles(samples_ratio, profiles_manager.profiles_records, profiles_manager.lost_profiles_records)

        if evaluate_models:
            IPC_evaluation = IPC_model.evaluate(x_test, uncertainty_test, models_metrics)
            APC_evaluation = APC_model.evaluate(x_test, uncertainty_test, models_metrics)
            results.set_models_evaluation(IPC_evaluation, APC_evaluation)

        return MPC_values, results


class Med3paDetectronResults:
    """
    Class to store and manage results from the MED3PA and Detectron method.
    """
    def __init__(self) -> None:
        self.detectron_res: DetectronResult = {}
        self.med3pa_reference: Med3paResults = {}
        self.med3pa_testing: Med3paResults = {}
        self.detectron_profiles: Dict = {}

    def set_results(self, detectron_results: DetectronResult, med3pa_ref: Med3paResults, med3pa_testing: Med3paResults, detectron_profiles: Dict) -> None:
        """
        Set results for Detectron and MED3PA.

        Args:
            detectron_results (DetectronResult): Results from the Detectron experiment.
            med3pa_ref (Med3paResults): Results from the MED3PA reference experiment.
            med3pa_testing (Med3paResults): Results from the MED3PA testing experiment.
            detectron_profiles (Dict): Dictionary of detectron profiles.
        """
        self.detectron_res = detectron_results
        self.med3pa_reference = med3pa_ref
        self.med3pa_testing = med3pa_testing
        self.detectron_profiles = detectron_profiles

    def save(self, file_path: str) -> None:
        """
        Save the detectron and med3pa results to JSON files.

        Args:
            file_path (str): The file path to save the JSON files.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)
        
        # Save detectron profiles
        with open(f'{file_path}/detectron_profiles.json', 'w') as file:
            json.dump(self.detectron_profiles, file, default=to_serializable, indent=4)
        
        # Save detectron results in a subfolder
        detectron_path = f'{file_path}/detectron_testing'
        os.makedirs(detectron_path, exist_ok=True)
        self.detectron_res.save(detectron_path)
        
        # Save med3pa reference in a subfolder
        med3pa_reference_path = f'{file_path}/3pa_reference'
        os.makedirs(med3pa_reference_path, exist_ok=True)
        self.med3pa_reference.save(med3pa_reference_path)
        
        # Save med3pa testing in a subfolder
        med3pa_testing_path = f'{file_path}/3pa_testing'
        os.makedirs(med3pa_testing_path, exist_ok=True)
        self.med3pa_testing.save(med3pa_testing_path)


class Med3paDetectron:
    @staticmethod
    def _filter_profiles(X: np.ndarray, Y_true: np.ndarray, predicted_prob: np.ndarray, Y_pred: np.ndarray, mpc_values: np.ndarray, path: List[str], min_confidence_level: float, features: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filters datasets based on a given path of conditions derived from a profile. This method allows for the selection of subsets of data corresponding to specific criteria defined in the path.

        Args:
            X: np.ndarray: The feature dataset from which rows are to be selected.
            Y_true: np.ndarray: The true labels corresponding to the dataset X.
            predicted_prob: np.ndarray: The predicted probabilities corresponding to the dataset X.
            Y_pred: np.ndarray: The predicted labels corresponding to the dataset X.
            mpc_values: np.ndarray: The mpc values corresponding to the dataset X.
            path: list: A list of conditions defining the path to filter by, with each condition formatted as "column_name operator value".
            min_confidence_level (float): float: Minimum confidence level to filter the data.
            features: list: List of feature names.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Filtered versions of X, Y_true, predicted_prob, Y_pred, and mpc_values based on the path conditions.

        Raises:
            ValueError: If an unsupported operator is included in any condition.
        """
        # Start with a mask that selects all rows
        mask = np.ones(len(X), dtype=bool)
        
        for condition in path:
            if condition == '*':
                continue  # Skip the root node indicator

            # Parse the condition string
            column_name, operator, value_str = condition.split(' ')
            column_index = features.index(column_name)  # Map feature name to index
            try:
                value = float(value_str)
            except ValueError:
                # If conversion fails, the string is not a number. Handle it appropriately.
                value = value_str  # If it's supposed to be a string, leave it as string
                        
            # Apply the condition to update the mask
            if operator == '>':
                mask &= X[:, column_index] > value
            elif operator == '<':
                mask &= X[:, column_index] < value
            elif operator == '>=':
                mask &= X[:, column_index] >= value
            elif operator == '<=':
                mask &= X[:, column_index] <= value
            elif operator == '==':
                mask &= X[:, column_index] == value
            elif operator == '!=':
                mask &= X[:, column_index] != value
            else:
                raise ValueError(f"Unsupported operator '{operator}' in condition '{condition}'.")

        # Filter the data according to the profile and min confidence level
        filtered_x = X[mask]
        filtered_y_true = Y_true[mask]
        filtered_prob = predicted_prob[mask] if predicted_prob is not None else None
        filtered_y_pred = Y_pred[mask] if Y_pred is not None else None
        filtered_mpc_values = mpc_values[mask] if mpc_values is not None else None


        filtered_x = filtered_x[filtered_mpc_values>=min_confidence_level]
        filtered_y_true = filtered_y_true[filtered_mpc_values>=min_confidence_level]
        filtered_prob = filtered_prob[filtered_mpc_values>=min_confidence_level] if predicted_prob is not None else None
        filtered_y_pred = filtered_y_pred[filtered_mpc_values>=min_confidence_level] if Y_pred is not None else None
        filtered_mpc_values = filtered_mpc_values[filtered_mpc_values>=min_confidence_level] if mpc_values is not None else None

        return filtered_x, filtered_y_true, filtered_prob, filtered_y_pred, filtered_mpc_values

    @staticmethod
    def detectron_by_profiles(datasets: DatasetsManager,
                              med3pa_testing_results: Med3paResults,
                              training_params: Dict,
                              base_model_manager: BaseModelManager,
                              testing_mpc_values: np.ndarray,
                              reference_mpc_values: np.ndarray,
                              samples_size: int = 20,
                              ensemble_size: int = 10,
                              num_calibration_runs: int = 100,
                              patience: int = 3,
                              significance_level: float = 0.05, 
                              test_strategy: Type[DisagreementStrategy_z_mean] = DisagreementStrategy_z_mean,
                              allow_margin: bool = False, 
                              margin: float = 0.05, 
                              all_dr: bool = True) -> Dict:
        """Runs the Detectron method for different profiles.

        Args:
            datasets (DatasetsManager): The datasets manager instance.
            med3pa_testing_results (Med3paResults): The results from the MED3PA testing experiment.
            training_params (dict): Parameters for training the models.
            base_model_manager (BaseModelManager): The base model manager instance.
            testing_mpc_values (np.ndarray): MPC values for the testing data.
            reference_mpc_values (np.ndarray): MPC values for the reference data.
            samples_size (int, optional): Sample size for the Detectron experiment, by default 20.
            ensemble_size (int, optional): Number of models in the ensemble, by default 10.
            num_calibration_runs (int, optional): Number of calibration runs, by default 100.
            patience (int, optional): Patience for early stopping, by default 3.
            significance_level (float, optional): Significance level for the test, by default 0.05.
            test_strategy (Type[DisagreementStrategy_z_mean], optional): The strategy for testing disagreement, by default DisagreementStrategy_z_mean.
            allow_margin (bool, optional): Whether to allow a margin in the test, by default False.
            margin (float, optional): Margin value for the test, by default 0.05.
            all_dr (bool, optional): Whether to run for all declaration rates, by default False.

        Returns:
            Dict: Dictionary of med3pa profiles with detectron results.
        """
        min_positive_ratio = min(k for k in med3pa_testing_results.profiles_records.keys() if k > 0)
        dr_dict = med3pa_testing_results.profiles_records[min_positive_ratio]
        last_min_confidence_level = 1.01     
        for dr, profiles in dr_dict.items():
            if not all_dr and dr != 100:
                continue  # Skip all dr values except the first one if all_dr is False

            sorted_accuracies = np.sort(testing_mpc_values)
            if dr == 0:
                min_confidence_level = 1.01
            else:
                min_confidence_level = sorted_accuracies[int(len(sorted_accuracies) * (1 - dr / 100))]
            if last_min_confidence_level != min_confidence_level:
                for profile in profiles:
                    detectron_results = {}
                    p_x, p_y_true, _, _, _ = Med3paDetectron._filter_profiles(
                        datasets.reference_set.features, datasets.reference_set.true_labels, datasets.reference_set.pseudo_probabilities, datasets.reference_set.pseudo_labels, reference_mpc_values, profile.path, min_confidence_level, datasets.column_labels)
                    q_x, q_y_true, _, _, _ = Med3paDetectron._filter_profiles(
                        datasets.testing_set.features, datasets.testing_set.true_labels, datasets.testing_set.pseudo_probabilities, datasets.testing_set.pseudo_labels, testing_mpc_values, profile.path, min_confidence_level, datasets.column_labels)
                    if len(p_y_true) != 0 and len(q_y_true) != 0:
                        if len(q_y_true) < samples_size: 
                            detectron_results['Executed'] = "Not enough test data"
                            detectron_results['data'] = [len(p_y_true), len(q_y_true)]                    
                        elif 2 * samples_size > len(p_y_true):
                            detectron_results['Executed'] = "Not enough calibration data"
                            detectron_results['data'] = [len(p_y_true), len(q_y_true)]
                        else:
                            profile_set = DatasetsManager()
                            tesing_set = MaskedDataset(q_x, q_y_true)
                            profile_set.testing_set = tesing_set
                            ref_set = MaskedDataset(p_x, p_y_true)
                            profile_set.reference_set = ref_set

                            profile_set.base_model_training_set = datasets.base_model_training_set
                            profile_set.base_model_validation_set = datasets.base_model_validation_set

                            experiment_det, _, _ = DetectronExperiment.run(
                                datasets=profile_set, training_params=training_params, base_model_manager=base_model_manager,
                                samples_size=samples_size, num_calibration_runs=num_calibration_runs, ensemble_size=ensemble_size,
                                patience=patience, test_strategy=test_strategy, significance_level=significance_level,
                                allow_margin=allow_margin, margin=margin)
                            
                            detectron_results = experiment_det.get_experiments_results(test_strategy, significance_level)
                            detectron_results['Executed'] = "Yes"
                            detectron_results['data'] = [len(p_y_true), len(q_y_true)]

                    else:
                        detectron_results['Executed'] = "Inexistant profile in calibration data"
                        detectron_results['data'] = [len(p_y_true), len(q_y_true)]

                    profile.update_detectron_results(detectron_results)
                
                last_profiles = [profile.to_dict() for profile in profiles]
                last_min_confidence_level = min_confidence_level
            else:
                for i in range(len(profiles)):
                    profiles[i] = last_profiles[i].copy()

        return dr_dict
    
    @staticmethod
    def run(datasets: DatasetsManager,
            training_params: Dict,
            base_model_manager: BaseModelManager,
            uncertainty_metric: Type[UncertaintyMetric],
            samples_size: int = 20,
            ensemble_size: int = 10,
            num_calibration_runs: int = 100,
            patience: int = 3,
            significance_level: float = 0.05, 
            test_strategy: Type[DisagreementStrategy_z_mean] = DisagreementStrategy_z_mean,
            allow_margin: bool = False, 
            margin: float = 0.05,
            ipc_type: Type[RegressionModel] = RandomForestRegressorModel,
            ipc_params: Dict = None,
            ipc_grid_params: Dict = None, 
            ipc_cv: int = None,
            apc_params: Dict = None,
            apc_grid_params: Dict = None,
            apc_cv: int = None,
            samples_ratio_min: int = 0,
            samples_ratio_max: int = 50, 
            samples_ratio_step: int = 5,
            med3pa_metrics: List[str] = ['Auc', 'Accuracy', 'BalancedAccuracy'],
            evaluate_models: bool = False,
            models_metrics: List[str] = ['MSE', 'RMSE'],
            all_dr: bool = False) -> Med3paDetectronResults:
        """Run the MED3PA and Detectron experiment.

        Args:
            datasets (DatasetsManager): The datasets manager instance.
            training_params (dict): Parameters for training the models.
            base_model_manager (BaseModelManager): The base model manager instance.
            uncertainty_metric (Type[UncertaintyMetric]): The uncertainty metric to use.
            samples_size (int, optional): Sample size for the Detectron experiment, by default 20.
            ensemble_size (int, optional): Number of models in the ensemble, by default 10.
            num_calibration_runs (int, optional): Number of calibration runs, by default 100.
            patience (int, optional): Patience for early stopping, by default 3.
            significance_level (float, optional): Significance level for the test, by default 0.05.
            test_strategy (Type[DisagreementStrategy_z_mean], optional): The strategy for testing disagreement, by default DisagreementStrategy_z_mean.
            allow_margin (bool, optional): Whether to allow a margin in the test, by default False.
            margin (float, optional): Margin value for the test, by default 0.05.
            ipc_type (Type[RegressionModel], optional): The class of the regressor model to use for IPC, by default RandomForestRegressorModel.
            ipc_params (dict, optional): Parameters for initializing the IPC regressor model, by default None.
            ipc_grid_params (dict, optional): Grid search parameters for optimizing the IPC model, by default None.
            ipc_cv (int, optional): Number of cross-validation folds for optimizing the IPC model, by default None.
            apc_params (dict, optional): Parameters for initializing the APC regressor model, by default None.
            apc_grid_params (dict, optional): Grid search parameters for optimizing the APC model, by default None.
            apc_cv (int, optional): Number of cross-validation folds for optimizing the APC model, by default None.
            samples_ratio_min (int, optional): Minimum sample ratio, by default 0.
            samples_ratio_max (int, optional): Maximum sample ratio, by default 50.
            samples_ratio_step (int, optional): Step size for sample ratio, by default 5.
            med3pa_metrics (list of str, optional): List of metrics to calculate, by default ['Auc', 'Accuracy', 'BalancedAccuracy'].
            evaluate_models (bool, optional): Whether to evaluate the models, by default False.
            models_metrics (list of str, optional): List of metrics for model evaluation, by default ['MSE', 'RMSE'].
            all_dr (bool, optional): Whether to run for all declaration rates, by default False.

        Returns:
            Med3paDetectronResults: Results of the MED3PA and Detectron experiment.
        """
        reference_x, reference_y = datasets.get_reference_data()
        testing_x, testing_y = datasets.get_testing_data()
        features = datasets.column_labels
        reference_mpc_values, reference_3pa_res = Med3paExperiment.run(x=reference_x, y_true=reference_y, features=features,
                                                base_model_manager=base_model_manager, uncertainty_metric=uncertainty_metric, predicted_probabilities=None,
                                                ipc_params=ipc_params, ipc_grid_params=ipc_grid_params, ipc_cv=ipc_cv, ipc_type=ipc_type,
                                                apc_params=apc_params, apc_grid_params=apc_grid_params, apc_cv=apc_cv,
                                                evaluate_models=evaluate_models, models_metrics=models_metrics,
                                                samples_ratio_min=samples_ratio_min, samples_ratio_max=samples_ratio_max, samples_ratio_step=samples_ratio_step,
                                                med3pa_metrics=med3pa_metrics)
        
        testing_mpc_values, testing_3pa_res = Med3paExperiment.run(x=testing_x, y_true=testing_y, features=features,
                                                base_model_manager=base_model_manager, uncertainty_metric=uncertainty_metric, predicted_probabilities=None,
                                                ipc_params=ipc_params, ipc_grid_params=ipc_grid_params, ipc_cv=ipc_cv, ipc_type=ipc_type,
                                                apc_params=apc_params, apc_grid_params=apc_grid_params, apc_cv=apc_cv,
                                                evaluate_models=evaluate_models, models_metrics=models_metrics,
                                                samples_ratio_min=samples_ratio_min, samples_ratio_max=samples_ratio_max, samples_ratio_step=samples_ratio_step,
                                                med3pa_metrics=med3pa_metrics)
        
                        

        detectron_results , _, _ = DetectronExperiment.run(datasets=datasets, training_params=training_params, base_model_manager=base_model_manager,
                                                    samples_size=samples_size, num_calibration_runs=num_calibration_runs, ensemble_size=ensemble_size,
                                                    patience=patience, test_strategy=test_strategy, significance_level=significance_level,
                                                    allow_margin=allow_margin, margin=margin)
        
        detectron_profiles_res = Med3paDetectron.detectron_by_profiles(datasets=datasets, med3pa_testing_results=testing_3pa_res,training_params=training_params, base_model_manager=base_model_manager,
                                                                        testing_mpc_values=testing_mpc_values, reference_mpc_values=reference_mpc_values,
                                                                        samples_size=samples_size, num_calibration_runs=num_calibration_runs, ensemble_size=ensemble_size,
                                                                        patience=patience, test_strategy=test_strategy, significance_level=significance_level,
                                                                        allow_margin=allow_margin, margin=margin, all_dr=all_dr)
        
        results = Med3paDetectronResults()
        results.set_results(detectron_results=detectron_results, med3pa_ref= reference_3pa_res, med3pa_testing=testing_3pa_res, detectron_profiles=detectron_profiles_res)
        return results
