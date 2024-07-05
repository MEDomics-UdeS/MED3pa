"""
Orchestrates the execution of the med3pa method and integrates the functionality of other modules to run comprehensive experiments. 
It includes classes to manage and store results ``Med3paResults``, execute experiments like ``Med3paExperiment`` and ``Med3paDetectronExperiment``, and integrate results from the Detectron method ``Med3paDetectronResults``
"""
import json
import os
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from sklearn.model_selection import train_test_split

from MED3pa.datasets import DatasetsManager, MaskedDataset
from MED3pa.detectron.experiment import DetectronExperiment, DetectronResult, DetectronStrategy, EnhancedDisagreementStrategy
from MED3pa.med3pa.mdr import MDRCalculator
from MED3pa.med3pa.models import *
from MED3pa.med3pa.profiles import Profile, ProfilesManager
from MED3pa.med3pa.uncertainty import *
from MED3pa.models.base import BaseModelManager
from MED3pa.models.classification_metrics import *
from MED3pa.models.concrete_regressors import *


def to_serializable(obj: Any, additional_arg: Any = None) -> Any:
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
        if additional_arg is not None:
            return obj.to_dict(additional_arg)
        else:
            return obj.to_dict()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj

class Med3paRecord:
    """
    Class to store and manage results from the MED3PA method on one set.
    """
    def __init__(self) -> None:
        self.metrics_by_dr: Dict[int, Dict] = {}
        self.models_evaluation: Dict[str, Dict] = {}
        self.profiles_manager: ProfilesManager = None
        self.datasets: Dict[int, MaskedDataset] = {}
        self.experiment_config = {}
    
    def set_metrics_by_dr(self, metrics_by_dr: Dict) -> None:
        """
        Set the calculated metrics by declaration rate.

        Args:
            metrics_by_dr (Dict): Dictionary of metrics by declaration rate.
        """
        self.metrics_by_dr = metrics_by_dr
    
    def set_profiles_manager(self, profile_manager : ProfilesManager) -> None:
        """
        Set the profile manager for this Med3paResults instance.
 
        Args:
            profile_manager (ProfilesManager): The ProfileManager instance.
        """
        self.profiles_manager = profile_manager

    def set_models_evaluation(self, ipc_evaluation: Dict, apc_evaluation: Dict=None) -> None:
        """
        Set models evaluation metrics.

        Args:
            ipc_evaluation (Dict): Evaluation metrics for IPC model.
            apc_evaluation (Dict): Evaluation metrics for APC model.
        """
        self.models_evaluation['IPC_evaluation'] = ipc_evaluation
        
        if apc_evaluation is not None:
            self.models_evaluation['APC_evaluation'] = apc_evaluation
    
    def set_dataset(self, samples_ratio: int, dataset: MaskedDataset) -> None:
        """
        Saves the dataset for a given sample ratio.

        Args:
            samples_ratio (int): The sample ratio.
            dataset (MaskedDataset): The MaskedDataset instance.
        """

        self.datasets[samples_ratio] = dataset

    def save(self, file_path: str) -> None:
        """
        Saves the experiment results.

        Args:
            file_path (str): The file path to save the JSON files.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)

        with open(f'{file_path}/metrics_dr.json', 'w') as file:
            json.dump(self.metrics_by_dr, file, default=to_serializable, indent=4)
         
        if self.profiles_manager is not None:
            with open(f'{file_path}/profiles.json', 'w') as file:
                json.dump(self.profiles_manager.get_profiles(), file, default=to_serializable, indent=4)
            with open(f'{file_path}/lost_profiles.json', 'w') as file:
                json.dump(self.profiles_manager.get_lost_profiles(), file, default=lambda x: to_serializable(x, additional_arg = False), indent=4)
        
        if self.models_evaluation is not None:
            with open(f'{file_path}/models_evaluation.json', 'w') as file:
                json.dump(self.models_evaluation, file, default=to_serializable, indent=4)
        
        for samples_ratio, dataset in self.datasets.items():
            dataset_path = os.path.join(file_path, f'dataset_{samples_ratio}.csv')
            dataset.save_to_csv(dataset_path)

    def get_profiles_manager(self) -> ProfilesManager:
        """
        Retrieves the profiles manager for this Med3paResults instance
        """
        return self.profiles_manager


class Med3paResults:
    """
    Class to store and manage results from the MED3PA complete experiment.
    """
    def __init__(self, reference_record:Med3paRecord, test_record:Med3paRecord) -> None:
        self.reference_record = reference_record
        self.test_record = test_record
        self.experiment_config ={}

    def set_experiment_config(self, config: Dict[str, Any]) -> None:
        """
        Sets or updates the configuration for the Med3pa experiment.

        Args:
            config (Dict[str, Any]): A dictionary of experiment configuration.
        """
        self.experiment_config.update(config)

    def save(self, file_path: str) -> None:
        """
        Saves the experiment results.

        Args:
            file_path (str): The file path to save the JSON files.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)
        
        reference_path = f'{file_path}/reference/'
        test_path = f'{file_path}/test/'
        self.reference_record.save(file_path=reference_path)
        self.test_record.save(file_path=test_path)

        with open(f'{file_path}/experiment_config.json', 'w') as file:
            json.dump(self.experiment_config, file, default=to_serializable, indent=4)
        

class Med3paExperiment:
    """
    Class to run the MED3PA method experiment.
    """
    @staticmethod
    def run(datasets_manager: DatasetsManager,
            base_model_manager: BaseModelManager = None,
            uncertainty_metric: str = 'absolute_error',
            ipc_type: str = 'RandomForestRegressor',
            ipc_params: Dict = None,
            ipc_grid_params: Dict = None, 
            ipc_cv: int = 4,
            apc_params: Dict = None,
            apc_grid_params: Dict = None,
            apc_cv: int = 4,
            samples_ratio_min: int = 0,
            samples_ratio_max: int = 50, 
            samples_ratio_step: int = 5,
            med3pa_metrics: List[str] = [],
            evaluate_models: bool = False,
            mode: str = 'mpc',
            models_metrics: List[str] = ['MSE', 'RMSE']) -> Med3paResults:
        
        """Runs the MED3PA experiment on both reference and testing sets.

        Args:
            datasets_manager (DatasetsManager): the datasets manager containing the dataset to use in the experiment.
            base_model_manager (BaseModelManager, optional): Instance of BaseModelManager to get the base model, by default None.
            uncertainty_metric (str, optional): the uncertainty metric ysed to calculate uncertainty, by default absolute_error.
            ipc_type (str, optional): The regressor model to use for IPC, by default RandomForestRegressor.
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
            Med3paResults: the results of the MED3PA experiment on the reference set and testing set.
        """
        print("Running MED3pa Experiment on the reference set:")
        results_reference, ipc_config, apc_config = Med3paExperiment._run_by_set(datasets_manager=datasets_manager,set= 'reference',base_model_manager= base_model_manager, 
                                                         uncertainty_metric=uncertainty_metric,
                                                         ipc_type=ipc_type, ipc_params=ipc_params, ipc_grid_params=ipc_grid_params, ipc_cv=ipc_cv, 
                                                         apc_params=apc_params,apc_grid_params=apc_grid_params, apc_cv=apc_cv, 
                                                         samples_ratio_min=samples_ratio_min, samples_ratio_max=samples_ratio_max, samples_ratio_step=samples_ratio_step, 
                                                         med3pa_metrics=med3pa_metrics, evaluate_models=evaluate_models, models_metrics=models_metrics, mode=mode)
        print("Running MED3pa Experiment on the reference set:")
        results_testing, ipc_config, apc_config = Med3paExperiment._run_by_set(datasets_manager=datasets_manager,set= 'testing',base_model_manager= base_model_manager, 
                                                         uncertainty_metric=uncertainty_metric,
                                                         ipc_type=ipc_type, ipc_params=ipc_params, ipc_grid_params=ipc_grid_params, ipc_cv=ipc_cv, 
                                                         apc_params=apc_params,apc_grid_params=apc_grid_params, apc_cv=apc_cv, 
                                                         samples_ratio_min=samples_ratio_min, samples_ratio_max=samples_ratio_max, samples_ratio_step=samples_ratio_step, 
                                                         med3pa_metrics=med3pa_metrics, evaluate_models=evaluate_models, models_metrics=models_metrics, mode=mode)
        
        results = Med3paResults(results_reference, results_testing)
        experiment_config = {
            'experiment_name': "Med3paExperiment",
            'datasets':datasets_manager.get_info(),
            'base_model': base_model_manager.get_instance().get_info(),
            'uncertainty_metric': uncertainty_metric,
            'ipc_model': ipc_config,
            'apc_model': apc_config,
            'samples_ratio_min': samples_ratio_min,
            'samples_ratio_max': samples_ratio_max,
            'samples_ratio_step': samples_ratio_step,
            'med3pa_metrics': med3pa_metrics,
            'evaluate_models':evaluate_models,
            'models_evaluation_metrics': models_metrics,
            'mode':mode
        }
        results.set_experiment_config(experiment_config)

        return results
    
    @staticmethod
    def _run_by_set(datasets_manager: DatasetsManager,
            set: str = 'reference',
            base_model_manager: BaseModelManager = None,
            uncertainty_metric: str = 'absolute_error',
            ipc_type: str = 'RandomForestRegressor',
            ipc_params: Dict = None,
            ipc_grid_params: Dict = None, 
            ipc_cv: int = 4,
            apc_params: Dict = None,
            apc_grid_params: Dict = None,
            apc_cv: int = 4,
            samples_ratio_min: int = 0,
            samples_ratio_max: int = 50, 
            samples_ratio_step: int = 5,
            med3pa_metrics: List[str] = [],
            evaluate_models: bool = False,
            mode: str = 'mpc',
            models_metrics: List[str] = ['MSE', 'RMSE']) -> Tuple[Med3paRecord, dict, dict]:
        
        """Orchestrates the MED3PA experiment on one specific set of the dataset.

        Args:
            datasets_manager (DatasetsManager): the datasets manager containing the dataset to use in the experiment.
            base_model_manager (BaseModelManager, optional): Instance of BaseModelManager to get the base model, by default None.
            uncertainty_metric (str, optional): the uncertainty metric ysed to calculate uncertainty, by default absolute_error.
            ipc_type (str, optional): The regressor model to use for IPC, by default RandomForestRegressor.
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
            Med3paRecord: the results of the MED3PA experiment.
        """
        # retrieve the dataset based on the set type
        try:
            if set == 'reference':
                dataset = datasets_manager.get_dataset_by_type(dataset_type="reference", return_instance=True)
            elif set == 'testing':
                dataset = datasets_manager.get_dataset_by_type(dataset_type="testing", return_instance=True)
            else:
                raise ValueError("The set must be either the reference set or the testing set")
        except ValueError as e:  
            dataset = None

        if dataset is None:
            return None
        
        valid_modes = ['mpc', 'apc', 'ipc']

        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. The mode must be one of {valid_modes}.")
        
        # retrieve different dataset components to calculate the metrics
        x = dataset.get_observations()
        y_true = dataset.get_true_labels()
        predicted_probabilities = dataset.get_pseudo_probabilities()
        features = datasets_manager.get_column_labels()

        # Initialize base model and predict probabilities if not provided
        if base_model_manager is None and predicted_probabilities is None:
            raise ValueError("Either the base model or the predicted probabilities should be provided!")
        
        if predicted_probabilities is None:
            base_model = base_model_manager.get_instance()
            predicted_probabilities = base_model.predict(x, True)

        dataset.set_pseudo_labels(predicted_probabilities)

        # Calculate uncertainty values
        uncertainty_calc = UncertaintyCalculator(uncertainty_metric)
        uncertainty_values = uncertainty_calc.calculate_uncertainty(x, predicted_probabilities, y_true)

        # set predicted labels
        dataset.set_pseudo_probs_labels(predicted_probabilities, 0.5)
        
        if evaluate_models:
            x_train, x_test, uncertainty_train, uncertainty_test = train_test_split(x, uncertainty_values, test_size=0.1, random_state=42)
        else:
            x_train = x
            uncertainty_train = uncertainty_values

        if med3pa_metrics == []:
            med3pa_metrics = ClassificationEvaluationMetrics.supported_metrics()
        
        results = Med3paRecord()

        # Create and train IPCModel
        IPC_model = IPCModel(ipc_type, ipc_params)
        IPC_model.train(x_train, uncertainty_train)
        print("IPC Model training completed.")
        
        # optimize IPC model if grid params were provided
        if ipc_grid_params is not None:
            IPC_model.optimize(ipc_grid_params, ipc_cv, x_train, uncertainty_train)
            print("IPC Model optimization done.")

        # Predict IPC values
        IPC_values = IPC_model.predict(x)
        
        if mode in ['mpc', 'apc']:
            # Create and train APCModel
            APC_model = APCModel(features, apc_params)
            APC_model.train(x, IPC_values)
            print("APC Model training completed.")

            # optimize APC model if grid params were provided
            if apc_grid_params is not None:
                APC_model.optimize(apc_grid_params, apc_cv, x_train, uncertainty_train)
                print("APC Model optimization done.")

            profiles_manager = ProfilesManager(features)

            for samples_ratio in range(samples_ratio_min, samples_ratio_max + 1, samples_ratio_step):
                
                # Predict APC values
                APC_values = APC_model.predict(x, min_samples_ratio=samples_ratio)

                if mode == 'mpc':
                    # Create and predict MPC values
                    MPC_model = MPCModel(IPC_values=IPC_values, APC_values=APC_values)
                else:
                    MPC_model = MPCModel(APC_values=APC_values)

                MPC_values = MPC_model.predict()
                dataset.set_confidence_scores(MPC_values)
                
                print("Confidence scores calculated for minimum_samples_ratio = ", samples_ratio)

                # Calculate profiles and their metrics by declaration rate
                tree = APC_model.treeRepresentation
                MDRCalculator.calc_profiles(profiles_manager, tree, MPC_values, samples_ratio)
                MDRCalculator.calc_metrics_by_profiles(profiles_manager, datasets_manager, med3pa_metrics, set=set)

                cloned_dataset = dataset.clone()
                results.set_profiles_manager(profiles_manager)
                results.set_dataset(samples_ratio=samples_ratio, dataset=cloned_dataset)
                
                print("Results extracted for minimum_samples_ratio = ", samples_ratio)
        
        # Calculate metrics by declaration rate
        # Create and predict MPC values using only the IPC values
        MPC_model = MPCModel(IPC_values=IPC_values)
        MPC_values = MPC_model.predict()
        
        dataset.set_confidence_scores(MPC_values)
        metrics_by_dr = MDRCalculator.calc_metrics_by_dr(datasets_manager=datasets_manager, metrics_list=med3pa_metrics, set=set)
        results.set_metrics_by_dr(metrics_by_dr)

        # evaluate models
        if evaluate_models:
            if mode in ['mpc', 'apc']:
                IPC_evaluation = IPC_model.evaluate(x_test, uncertainty_test, models_metrics)
                APC_evaluation = APC_model.evaluate(x_test, uncertainty_test, models_metrics)
                results.set_models_evaluation(IPC_evaluation, APC_evaluation)
                ipc_config = IPC_model.get_info()
                apc_config = APC_model.get_info() 
            else :
                IPC_evaluation = IPC_model.evaluate(x_test, uncertainty_test, models_metrics)
                results.set_models_evaluation(IPC_evaluation, None)
                ipc_config = IPC_model.get_info()
                apc_config = None 
        
        return results, ipc_config, apc_config  


class Med3paDetectronExperiment:
    @staticmethod
    def run(datasets: DatasetsManager,
            base_model_manager: BaseModelManager,
            uncertainty_metric: str = 'absolute_error',
            training_params: Dict =None,
            samples_size: int = 20,
            samples_size_profiles: int = 10,
            ensemble_size: int = 10,
            num_calibration_runs: int = 100,
            patience: int = 3,
            test_strategies: Union[str, List[str]] = "enhanced_disagreement_strategy",
            allow_margin: bool = False, 
            margin: float = 0.05,
            ipc_type: str = 'RandomForestRegressor',
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
            mode: str = 'mpc',
            all_dr: bool = False) ->  Tuple[Med3paResults, Med3paResults, DetectronResult]:
        """Runs the MED3PA and Detectron experiment.

        Args:
            datasets (DatasetsManager): The datasets manager instance.
            training_params (dict): Parameters for training the models.
            base_model_manager (BaseModelManager): The base model manager instance.
            uncertainty_metric (str, optional): the uncertainty metric ysed to calculate uncertainty, by default absolute_error.
            samples_size (int, optional): Sample size for the Detectron experiment, by default 20.
            samples_size_profiles (int, optional): Sample size for Profiles Detectron experiment, by default 10.
            ensemble_size (int, optional): Number of models in the ensemble, by default 10.
            num_calibration_runs (int, optional): Number of calibration runs, by default 100.
            patience (int, optional): Patience for early stopping, by default 3.
            test_strategies (Union[str, List[str]): strategies for testing disagreement, by default enhanced_disagreement_strategies.
            allow_margin (bool, optional): Whether to allow a margin in the test, by default False.
            margin (float, optional): Margin value for the test, by default 0.05.
            ipc_type (str, optional): The regressor model to use for IPC, by default RandomForestRegressor.
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
            Tuple[Med3paResults, DetectronResult]: Results of MED3pa on reference and testing sets, plus Detectron Results.
        """
        
        valid_modes = ['mpc', 'apc']

        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. The mode must be one of {valid_modes}.")
        
        med3pa_results = Med3paExperiment.run(datasets_manager=datasets, 
                                                                base_model_manager=base_model_manager, uncertainty_metric=uncertainty_metric,
                                                                ipc_params=ipc_params, ipc_grid_params=ipc_grid_params, ipc_cv=ipc_cv, ipc_type=ipc_type,
                                                                apc_params=apc_params, apc_grid_params=apc_grid_params, apc_cv=apc_cv,
                                                                evaluate_models=evaluate_models, models_metrics=models_metrics,
                                                                samples_ratio_min=samples_ratio_min, samples_ratio_max=samples_ratio_max, samples_ratio_step=samples_ratio_step,
                                                                med3pa_metrics=med3pa_metrics, mode=mode)
            
        print("Running Global Detectron Experiment:")
        detectron_results = DetectronExperiment.run(datasets=datasets, training_params=training_params, base_model_manager=base_model_manager,
                                                    samples_size=samples_size, num_calibration_runs=num_calibration_runs, ensemble_size=ensemble_size,
                                                    patience=patience, allow_margin=allow_margin, margin=margin)
        detectron_results.analyze_results(test_strategies)
        
        print("Running Profiled Detectron Experiment:")
        detectron_profiles_res = MDRCalculator.detectron_by_profiles(datasets=datasets, profiles_manager=med3pa_results.test_record.get_profiles_manager(),training_params=training_params, 
                                                                     base_model_manager=base_model_manager,
                                                                     samples_size=samples_size_profiles, num_calibration_runs=num_calibration_runs, ensemble_size=ensemble_size,
                                                                     patience=patience, strategies=test_strategies,
                                                                     allow_margin=allow_margin, margin=margin, all_dr=all_dr)
        
        experiment_config = {
            'experiment_name': "Med3paDetectronExperiment",
            'additional_training_params': training_params,
            'profiles_samples_size': samples_size_profiles,
            'cdcs_ensemble_size': ensemble_size,
            'num_runs': num_calibration_runs,
            'patience': patience,
            'allow_margin': allow_margin,
            'margin': margin
        }

        med3pa_results.set_experiment_config(experiment_config)

        return med3pa_results, detectron_results

