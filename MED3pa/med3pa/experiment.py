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
from MED3pa.med3pa.tree import TreeRepresentation

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
        self.tree = {}

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

    def set_tree(self, tree:TreeRepresentation):
        """
        Sets the constructed tree
        """
        self.tree = tree

    def set_dataset(self, mode: str, dataset: MaskedDataset) -> None:
        """
        Saves the dataset for a given sample ratio.
        Args:
            samples_ratio (int): The sample ratio.
            dataset (MaskedDataset): The MaskedDataset instance.
        """

        self.datasets[mode] = dataset

    def save(self, file_path: str) -> None:
        """
        Saves the experiment results.
        Args:
            file_path (str): The file path to save the JSON files.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)

        metrics_file_path = os.path.join(file_path, 'metrics_dr.json')
        with open(metrics_file_path, 'w') as file:
            json.dump(self.metrics_by_dr, file, default=to_serializable, indent=4)

        if self.profiles_manager is not None:
            profiles_file_path = os.path.join(file_path, 'profiles.json')
            with open(profiles_file_path, 'w') as file:
                json.dump(self.profiles_manager.get_profiles(), file, default=to_serializable, indent=4)

            lost_profiles_file_path = os.path.join(file_path, 'lost_profiles.json')
            with open(lost_profiles_file_path, 'w') as file:
                json.dump(self.profiles_manager.get_lost_profiles(), file, default=lambda x: to_serializable(x, additional_arg=False), indent=4)

        if self.models_evaluation is not None:
            models_evaluation_file_path = os.path.join(file_path, 'models_evaluation.json')
            with open(models_evaluation_file_path, 'w') as file:
                json.dump(self.models_evaluation, file, default=to_serializable, indent=4)

        for mode, dataset in self.datasets.items():
            dataset_path = os.path.join(file_path, f'dataset_{mode}.csv')
            dataset.save_to_csv(dataset_path)

        if self.tree is not None: 
            tree_path = os.path.join(file_path, 'tree.json')
            self.tree.save_tree(tree_path)

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
        self.experiment_config = {}
        self.detectron_results = None

    def set_detectron_results(self, detectron_results: DetectronResult=None) -> None:
        """
        Sets the detectron results for the Med3paDetectron experiment.
        Args:
            detectron_results (DetectronResult): The structure holding the detectron results.
        """
        self.detectron_results = detectron_results

    def set_experiment_config(self, config: Dict[str, Any]) -> None:
        """
        Sets or updates the configuration for the Med3pa experiment.
        Args:
            config (Dict[str, Any]): A dictionary of experiment configuration.
        """
        self.experiment_config.update(config)

    def set_models(self, ipc_model: IPCModel, apc_model:APCModel = None):
        self.ipc_model = ipc_model
        if apc_model:
            self.apc_model = apc_model

    def save(self, file_path: str) -> None:
        """
        Saves the experiment results.
        Args:
            file_path (str): The file path to save the JSON files.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)

        reference_path = os.path.join(file_path, 'reference')
        test_path = os.path.join(file_path, 'test')
        detectron_path = os.path.join(file_path, 'detectron')

        self.reference_record.save(file_path=reference_path)
        self.test_record.save(file_path=test_path)
        if self.detectron_results is not None:
            self.detectron_results.save(file_path=detectron_path, save_config=False)

        experiment_config_path = os.path.join(file_path, 'experiment_config.json')
        with open(experiment_config_path, 'w') as file:
            json.dump(self.experiment_config, file, default=to_serializable, indent=4)

    def save_models(self, file_path: str, mode:str ='all') -> None:
        """
        Saves the experiment ipc and apc models as a .pkl files, alongside the tree structure for the test set.
        Args:
            file_path (str): The file path to save the pickled files.
            mode (str): Defines the type of models to save, either ipc, apc, or both.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)
        
        if mode=='all':
            if self.ipc_model:
                ipc_path = os.path.join(file_path, 'ipc_model.pkl')
                self.ipc_model.save_model(ipc_path)
            if self.apc_model:
                apc_path = os.path.join(file_path, 'apc_model.pkl')
                self.apc_model.save_model(apc_path)
            tree_structure = self.test_record.tree
            tree_structure_path = os.path.join(file_path, 'tree.json')
            if tree_structure:
                tree_structure.save_tree(tree_structure_path)
        elif mode=='ipc':
            if self.ipc_model:
                ipc_path = os.path.join(file_path, 'ipc_model.pkl')
                self.ipc_model.save_model(ipc_path)
        elif mode=='apc':
            if self.apc_model:
                apc_path = os.path.join(file_path, 'apc_model.pkl')
                self.apc_model.save_model(apc_path)

            tree_structure = self.test_record.tree
            tree_structure_path = os.path.join(file_path, 'tree.json')
            if tree_structure:
                tree_structure.save_tree(tree_structure_path)

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
            pretrained_ipc: str = None,
            apc_params: Dict = None,
            apc_grid_params: Dict = None,
            apc_cv: int = 4,
            fixed_tree:str = None,
            pretrained_apc: str = None,
            samples_ratio_min: int = 0,
            samples_ratio_max: int = 50, 
            samples_ratio_step: int = 5,
            med3pa_metrics: List[str] = [],
            evaluate_models: bool = False,
            use_ref_models: bool = False, 
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
            pretrained_ipc (str, optional): path to a pretrained ipc, by default None.
            apc_params (dict, optional): Parameters for initializing the APC regressor model, by default None.
            apc_grid_params (dict, optional): Grid search parameters for optimizing the APC model, by default None.
            apc_cv (int, optional): Number of cross-validation folds for optimizing the APC model, by default None.
            fixed_tree (int, optional): a tree structure to use, by default None.
            pretrained_apc (str, optional): path to a pretrained apc, by default None.
            use_ref_models (bool, optional): whether or not to use the trained IPC and APC models from the reference set on the test set.
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
                                                         uncertainty_metric=uncertainty_metric, fixed_tree=fixed_tree,
                                                         ipc_type=ipc_type, ipc_params=ipc_params, ipc_grid_params=ipc_grid_params, ipc_cv=ipc_cv, pretrained_ipc=pretrained_ipc,
                                                         apc_params=apc_params,apc_grid_params=apc_grid_params, apc_cv=apc_cv, pretrained_apc=pretrained_apc,
                                                         samples_ratio_min=samples_ratio_min, samples_ratio_max=samples_ratio_max, samples_ratio_step=samples_ratio_step, 
                                                         med3pa_metrics=med3pa_metrics, evaluate_models=evaluate_models, models_metrics=models_metrics, mode=mode)
        print("Running MED3pa Experiment on the test set:")
        if use_ref_models:
            results_testing, ipc_config, apc_config = Med3paExperiment._run_by_set(datasets_manager=datasets_manager,set= 'testing',base_model_manager= base_model_manager, 
                                                            uncertainty_metric=uncertainty_metric, fixed_tree=fixed_tree,
                                                            ipc_type=ipc_type, ipc_params=ipc_params, ipc_grid_params=ipc_grid_params, ipc_cv=ipc_cv, pretrained_ipc=pretrained_ipc, ipc_instance=ipc_config,
                                                            apc_params=apc_params,apc_grid_params=apc_grid_params, apc_cv=apc_cv, pretrained_apc=pretrained_apc, apc_instance=apc_config,
                                                            samples_ratio_min=samples_ratio_min, samples_ratio_max=samples_ratio_max, samples_ratio_step=samples_ratio_step, 
                                                            med3pa_metrics=med3pa_metrics, evaluate_models=evaluate_models, models_metrics=models_metrics, mode=mode)
        else:
            results_testing, ipc_config, apc_config = Med3paExperiment._run_by_set(datasets_manager=datasets_manager,set= 'testing',base_model_manager= base_model_manager, 
                                                            uncertainty_metric=uncertainty_metric, fixed_tree=fixed_tree,
                                                            ipc_type=ipc_type, ipc_params=ipc_params, ipc_grid_params=ipc_grid_params, ipc_cv=ipc_cv, pretrained_ipc=pretrained_ipc, ipc_instance=None,
                                                            apc_params=apc_params,apc_grid_params=apc_grid_params, apc_cv=apc_cv, pretrained_apc=pretrained_apc, apc_instance=None,
                                                            samples_ratio_min=samples_ratio_min, samples_ratio_max=samples_ratio_max, samples_ratio_step=samples_ratio_step, 
                                                            med3pa_metrics=med3pa_metrics, evaluate_models=evaluate_models, models_metrics=models_metrics, mode=mode)

        results = Med3paResults(results_reference, results_testing)
        med3pa_params = {
            'uncertainty_metric': uncertainty_metric,
            'samples_ratio_min': samples_ratio_min,
            'samples_ratio_max': samples_ratio_max,
            'samples_ratio_step': samples_ratio_step,
            'med3pa_metrics': med3pa_metrics,
            'evaluate_models':evaluate_models,
            'models_evaluation_metrics': models_metrics,
            'mode':mode

        }
        experiment_config = {
            'experiment_name': "Med3paExperiment",
            'datasets':datasets_manager.get_info(),
            'base_model': base_model_manager.get_instance().get_info(),
            'ipc_model': ipc_config.get_info(),
            'apc_model': apc_config.get_info(),
            'experiment_params': med3pa_params
        }
        results.set_experiment_config(experiment_config)
        results.set_models(ipc_config, apc_config)
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
            pretrained_ipc:str = None,
            ipc_instance: IPCModel = None,
            apc_params: Dict = None,
            apc_grid_params: Dict = None,
            apc_cv: int = 4,
            apc_instance: APCModel = None,
            fixed_tree:str = None,
            pretrained_apc:str = None,
            samples_ratio_min: int = 0,
            samples_ratio_max: int = 50, 
            samples_ratio_step: int = 5,
            med3pa_metrics: List[str] = [],
            evaluate_models: bool = False,
            mode: str = 'mpc',
            models_metrics: List[str] = ['MSE', 'RMSE']) -> Tuple[Med3paRecord, dict, dict]:

        """
        Orchestrates the MED3PA experiment on one specific set of the dataset.
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

        # Step 1 : datasets and base model setting

        # Retrieve the dataset based on the set type
        if set == 'reference':
            dataset = datasets_manager.get_dataset_by_type(dataset_type="reference", return_instance=True)
        elif set == 'testing':
            dataset = datasets_manager.get_dataset_by_type(dataset_type="testing", return_instance=True)
        else:
            raise ValueError("The set must be either the reference set or the testing set")

        # retrieve different dataset components needed for the experiment
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

        dataset.set_pseudo_probs_labels(predicted_probabilities, 0.5)

        # Step 2 : Mode and metrics setup
        valid_modes = ['mpc', 'apc', 'ipc']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. The mode must be one of {valid_modes}.")

        if med3pa_metrics == []:
            med3pa_metrics = ClassificationEvaluationMetrics.supported_metrics()

        # Step 3 : Calculate uncertainty values
        uncertainty_calc = UncertaintyCalculator(uncertainty_metric)
        uncertainty_values = uncertainty_calc.calculate_uncertainty(x, predicted_probabilities, y_true)

        # Step 4: Set up splits to evaluate the models
        if evaluate_models:
            _, x_test, _, uncertainty_test = train_test_split(x, uncertainty_values, test_size=0.1, random_state=42)

        x_train = x
        uncertainty_train = uncertainty_values



        results = Med3paRecord()

        # Step 5: Create and train IPCModel
        if pretrained_ipc is None and ipc_instance is None:
            IPC_model = IPCModel(model_name=ipc_type, params=ipc_params, pretrained_model=None)
            IPC_model.train(x_train, uncertainty_train)
            print("IPC Model training complete.")
            # optimize IPC model if grid params were provided
            if ipc_grid_params is not None:
                IPC_model.optimize(ipc_grid_params, ipc_cv, x_train, uncertainty_train)
                print("IPC Model optimization complete.")
        elif pretrained_ipc is not None:
            IPC_model = IPCModel(model_name=ipc_type, params=ipc_params, pretrained_model=pretrained_ipc)
            print("Loaded a pretrained IPC model.")
        else:
            IPC_model = ipc_instance
            print("Used a trained IPC instance.")


        # Predict IPC values
        IPC_values = IPC_model.predict(x)
        print("Individualized confidence scores calculated.")

        if mode in ['mpc', 'apc']:

            # Step 6: Create and train APCModel
            if pretrained_apc is None and apc_instance is None:
                APC_model = APCModel(features=features, params=apc_params, tree_file_path=fixed_tree)
                APC_model.train(x, IPC_values)
                print("APC Model training complete.")
                # optimize APC model if grid params were provided
                if apc_grid_params is not None:
                    APC_model.optimize(apc_grid_params, apc_cv, x_train, uncertainty_train)
                    print("APC Model optimization complete.")
            elif pretrained_apc is not None:
                APC_model = APCModel(features=features, params=apc_params, pretrained_model=pretrained_apc)
                APC_model.train(x, IPC_values)
                print("Loaded a pretrained APC model.")
            else:
                APC_model = apc_instance
                print("Used a trainde IPC instance.")

            # Predict APC values
            APC_values = APC_model.predict(x)
            print("Aggregated confidence scores calculated.")
            # Save the tree structure created by the APCModel
            tree = APC_model.treeRepresentation
            results.set_tree(tree=tree)
            # Save the calculated confidence scores by the APCmodel
            dataset.set_confidence_scores(APC_values)
            cloned_dataset = dataset.clone()
            results.set_dataset(mode="apc", dataset=cloned_dataset)

            # Step 7: Create and train MPCModel
            if mode == 'mpc':
                # Create and predict MPC values
                MPC_model = MPCModel(IPC_values=IPC_values, APC_values=APC_values)
                MPC_values = MPC_model.predict()
                # Save the calculated confidence scores by the MPCmodel
                dataset.set_confidence_scores(MPC_values)
                cloned_dataset = dataset.clone()
                results.set_dataset(mode="mpc", dataset=cloned_dataset)
            else:
                MPC_model = MPCModel(APC_values=APC_values)
                MPC_values = MPC_model.predict()

            print("Mixed confidence scores calculated.")

            # Step 8: Calculate the profiles for the different samples_ratio and drs
            profiles_manager = ProfilesManager(features)
            for samples_ratio in range(samples_ratio_min, samples_ratio_max + 1, samples_ratio_step):

                # Calculate profiles and their metrics by declaration rate                
                MDRCalculator.calc_profiles(profiles_manager, tree, datasets_manager, MPC_values, samples_ratio, set=set)                
                MDRCalculator.calc_metrics_by_profiles(profiles_manager, datasets_manager, MPC_values, samples_ratio, med3pa_metrics, set=set)
                results.set_profiles_manager(profiles_manager)                
                print("Results extracted for minimum_samples_ratio = ", samples_ratio)


        # Calculate metrics by declaration rate
        # Create and predict MPC values using only the IPC values
        MPC_model = MPCModel(IPC_values=IPC_values)
        MPC_values = MPC_model.predict()
        # Save the confidence scores predicted by the IPCModel
        dataset.set_confidence_scores(MPC_values)
        cloned_dataset = dataset.clone()
        results.set_dataset(mode="ipc", dataset=cloned_dataset)
        metrics_by_dr = MDRCalculator.calc_metrics_by_dr(datasets_manager=datasets_manager, confidence_scores=MPC_values, metrics_list=med3pa_metrics, set=set)
        results.set_metrics_by_dr(metrics_by_dr)

        if mode in ['mpc', 'apc']:
            ipc_config = IPC_model
            apc_config = APC_model 
            if evaluate_models:
                IPC_evaluation = IPC_model.evaluate(x_test, uncertainty_test, models_metrics)
                APC_evaluation = APC_model.evaluate(x_test, uncertainty_test, models_metrics)
                results.set_models_evaluation(IPC_evaluation, APC_evaluation)
        else:
            ipc_config = IPC_model
            apc_config = None
            if evaluate_models:
                IPC_evaluation = IPC_model.evaluate(x_test, uncertainty_test, models_metrics)
                results.set_models_evaluation(IPC_evaluation, None)

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
            pretrained_ipc: str = None,
            apc_params: Dict = None,
            apc_grid_params: Dict = None,
            apc_cv: int = None,
            fixed_tree:str = None,
            pretrained_apc: str = None,
            samples_ratio_min: int = 0,
            samples_ratio_max: int = 50, 
            samples_ratio_step: int = 5,
            med3pa_metrics: List[str] = [],
            evaluate_models: bool = False,
            use_ref_models: bool = False, 
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
            pretrained_ipc (str, optional): path to a pretrained ipc, by default None.
            apc_params (dict, optional): Parameters for initializing the APC regressor model, by default None.
            apc_grid_params (dict, optional): Grid search parameters for optimizing the APC model, by default None.
            fixed_tree (int, optional): a tree structure to use, by default None.
            pretrained_apc (str, optional): path to a pretrained apc, by default None.
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
                                                                ipc_params=ipc_params, ipc_grid_params=ipc_grid_params, ipc_cv=ipc_cv, ipc_type=ipc_type, pretrained_ipc=pretrained_ipc,
                                                                apc_params=apc_params, apc_grid_params=apc_grid_params, apc_cv=apc_cv, fixed_tree=fixed_tree, pretrained_apc=pretrained_apc,
                                                                evaluate_models=evaluate_models, models_metrics=models_metrics,
                                                                samples_ratio_min=samples_ratio_min, samples_ratio_max=samples_ratio_max, samples_ratio_step=samples_ratio_step,
                                                                med3pa_metrics=med3pa_metrics, mode=mode, use_ref_models=use_ref_models)

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

        med3pa_detectron_params = {
            'uncertainty_metric': uncertainty_metric,
            'samples_ratio_min': samples_ratio_min,
            'samples_ratio_max': samples_ratio_max,
            'samples_ratio_step': samples_ratio_step,
            'med3pa_metrics': med3pa_metrics,
            'evaluate_models':evaluate_models,
            'models_evaluation_metrics': models_metrics,
            'mode':mode,
            'samples_size': samples_size,
            'profiles_samples_size': samples_size_profiles,
            'cdcs_ensemble_size': ensemble_size,
            'num_runs': num_calibration_runs,
            'patience': patience,
            'allow_margin': allow_margin,
            'margin': margin,
            'additional_training_params': training_params,

        }

        experiment_config = {
            'experiment_name': "Med3paDetectronExperiment",
            'experiment_params': med3pa_detectron_params,            
        }

        med3pa_results.set_detectron_results(detectron_results)
        med3pa_results.set_experiment_config(experiment_config)

        return med3pa_results