"""
This module encapsulates the execution logic for the Detectron method, managing the orchestration of the entire pipeline. It includes the ``DetectronExperiment`` abstract class, 
which outlines the protocol for setting up and running experiments. 
Additionally, the ``DetectronResult`` class is responsible for storing and managing the outcomes of these experiments, 
providing methods to access and analyze the trajectories and outcomes of this method's evaluation.
"""
from __future__ import annotations
from typing import Union, List, Type

import json
import os
from warnings import warn

from .ensemble import BaseModelManager, DatasetsManager, DetectronEnsemble
from .record import DetectronRecordsManager
from .strategies import *


class DetectronResult:
    """
    A class to store the results of a Detectron test.
    """

    strategy_mapping = {
        'original_disagreement_strategy': OriginalDisagreementStrategy,
        'mannwhitney_strategy': MannWhitneyStrategy,
        'enhanced_disagreement_strategy': EnhancedDisagreementStrategy
    }

    def __init__(self, cal_record: DetectronRecordsManager, test_record: DetectronRecordsManager):
        """
        Initializes the DetectronResult with calibration and test records.

        Args:
            cal_record (DetectronRecordsManager): Manager storing the results of running the Detectron on the 'reference' set.
            test_record (DetectronRecordsManager): Manager storing the results of running the Detectron on the 'testing' set.
        """
        self.cal_record = cal_record
        self.test_record = test_record
        self.test_results = []
        self.experiment_config = {}

    def calibration_trajectories(self):
        """
        Retrieves the results for each run and each model in the ensemble from the reference set.

        Returns:
            DataFrame: A DataFrame containing seed, model_id, and rejection_rate from the calibration records.
        """
        rec = self.cal_record.get_record()
        return rec[['seed', 'model_id', 'rejection_rate']]

    def test_trajectories(self):
        """
        Retrieves the results for each run and each model in the ensemble from the testing set.

        Returns:
            DataFrame: A DataFrame containing seed, model_id, and rejection_rate from the test records.
        """
        rec = self.test_record.get_record()
        return rec[['seed', 'model_id', 'rejection_rate']]

    def get_experiments_results(self):
        """
        Executes the Detectron tests using the specified strategy and records.

        Returns:
            dict: Results from executing the Detectron test.
        """
        return self.test_results

    def analyze_results(self, strategies: Union[str, List[str]]= ["enhanced_disagreement_strategy", "mannwhitney_strategy", "original_disagreement_strategy"]) -> list:
        """
        Appends the results of the Detectron tests for each strategy to self.test_results.

        Args:
            strategies (Union[str, List[str]]): Strategy name or list of strategy names.

        Returns:
            list: Updated list containing results for each strategy.
        """

        # Ensure strategies is a list of strategy names
        if isinstance(strategies, str):
            strategies = [strategies]  # Convert single strategy name to list

        self.experiment_config['experiment_params']['test_strategies'] = strategies

        for strategy_name in strategies:
            if strategy_name not in self.strategy_mapping:
                raise ValueError(f"Unrecognized strategy name: {strategy_name}. Available strategies: {list(self.strategy_mapping.keys())}")

            strategy_class = self.strategy_mapping[strategy_name]
            strategy_results = strategy_class.execute(self.cal_record, self.test_record)
            strategy_results['Strategy'] = strategy_name
            self.test_results.append(strategy_results)

        return self.test_results
    
    def set_experiment_config(self, config: dict):
        """
        Sets or updates the configuration for the Detectron experiment.

        Args:
            config (dict): A dictionary of hyperparameters used in the experiment.
        """
        self.experiment_config.update(config)

    def save(self, file_path: str, file_name: str = 'detectron_results', save_config=True):
        """
        Saves the Detectron results to JSON format.

        Args:
            file_path (str): The file path where the results should be saved.
            file_name (str): The file name.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)
        
        file_name_path = os.path.join(file_path, f'{file_name}.json')
        with open(file_name_path, 'w') as file:
            json.dump(self.test_results, file, indent=4)
        
        counts_dict = {}
        counts_dict['reference'] = self.cal_record.rejected_counts().tolist()
        counts_dict['test'] = self.test_record.rejected_counts().tolist()
        
        file_name_path_counts = os.path.join(file_path, 'rejection_counts.json')
        with open(file_name_path_counts, 'w') as file:
            json.dump(counts_dict, file, indent=4)

        if save_config:
            config_file_path = os.path.join(file_path, 'experiment_config.json')
            with open(config_file_path, 'w') as file:
                json.dump(self.experiment_config, file, indent=4)

    @classmethod
    def get_supported_strategies(cls) -> list:
        """
        Returns a list of supported strategy names.

        Returns:
            list: A list of strings representing the names of the supported strategies.
        """
        return list(cls.strategy_mapping.keys())
    
class DetectronExperiment:
    """
    Abstract base class that defines the protocol for running Detectron experiments.

    Methods:
        run: Orchestrates the entire process of a Detectron experiment using specified parameters and strategies.
    """
    @staticmethod
    def run(datasets: DatasetsManager,
            base_model_manager: BaseModelManager,
            training_params: dict=None,
            samples_size : int = 20,
            calib_result:DetectronRecordsManager=None,
            ensemble_size=10,
            num_calibration_runs=100,
            patience=3,
            allow_margin : bool = False, 
            margin = 0.05):
        """
        Orchestrates the process of a Detectron experiment, including ensemble training and testing, and strategy evaluation.

        Args:
            datasets (DatasetsManager): Manages the datasets used in the experiment.
            training_params (dict): Parameters for training the cdcs within the ensembles.
            base_model_manager (BaseModelManager): Manager for the base model operations.
            samples_size (int): Number of samples to use in each Detectron run. Defaults to 20.
            calib_result (Optional[DetectronRecordsManager]): Calibration results, if provided. Defaults to None.
            ensemble_size (int): Number of models in each ensemble. Defaults to 10.
            num_calibration_runs (int): Number of calibration runs. Defaults to 100.
            patience (int): Number of iterations with no improvement before stopping. Defaults to 3.
            allow_margin (bool): Allow a margin of error when comparing model outputs. Defaults to False.
            margin (float): Threshold for considering differences significant when margin is allowed. Defaults to 0.05.

        Returns:
            tuple: A tuple containing the Detectron results, experimental strategy results, and Detectron evaluation results, if conducted.
        """
        # create a calibration ensemble
        calibration_ensemble = DetectronEnsemble(base_model_manager, ensemble_size)
        
        # create a testing ensemble
        testing_ensemble = DetectronEnsemble(base_model_manager, ensemble_size)
        
        # ensure the reference set is larger compared to testing set
        reference_set = datasets.get_dataset_by_type(dataset_type="reference", return_instance=True)
        if reference_set is not None:
            test_size = len(reference_set)
            assert test_size > samples_size, \
                "The reference set must be larger than the testing set to perform statistical bootstrapping"
            if test_size < 2 * samples_size:
                warn("The reference set is smaller than twice the testing set, this may lead to poor calibration")
            if calib_result is not None:
                print("Calibration record on reference set provided, skipping Detectron execution on reference set.")
                cal_record = calib_result
            else:
            
            # evaluate the calibration ensemble
                cal_record = calibration_ensemble.evaluate_ensemble(datasets=datasets, 
                                                                    n_runs=num_calibration_runs,
                                                                    samples_size=samples_size, 
                                                                    training_params=training_params, 
                                                                    set='reference', 
                                                                    patience=patience, 
                                                                    allow_margin=allow_margin,
                                                                    margin=margin)
                print("Detectron execution on reference set completed.")
            
            test_record = testing_ensemble.evaluate_ensemble(datasets=datasets, 
                                                            n_runs=num_calibration_runs, 
                                                            samples_size=samples_size, 
                                                            training_params=training_params,
                                                            set='testing', 
                                                            patience=patience,
                                                            allow_margin=allow_margin,
                                                            margin=margin)
            print("Detectron execution on testing set completed.")


            assert cal_record.sample_size == test_record.sample_size, \
                "The calibration record must have been generated with the same sample size as the observation set"
            
    
        # save the detectron runs results
        detectron_results = DetectronResult(cal_record, test_record)
        detectron_params = {
            'additional_training_params': training_params,
            'samples_size': samples_size,
            'cdcs_ensemble_size': ensemble_size,
            'num_runs': num_calibration_runs,
            'patience': patience,
            'allow_margin': allow_margin,
            'margin': margin
        }
        experiment_config = {
            'experiment_name': "DetectronExperiment",
            'datasets':datasets.get_info(),
            'base_model': base_model_manager.get_instance().get_info(),
            'experiment_params': detectron_params
            
        }
        detectron_results.set_experiment_config(experiment_config)
        # return the Detectron results
        return detectron_results



