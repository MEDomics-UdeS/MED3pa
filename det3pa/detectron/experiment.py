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
    A class to store the results of a Detectron test
    """

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
    
    
    def analyze_results(self, strategies: Union[Type[DetectronStrategy], List[Type[DetectronStrategy]]]) -> list:
        """
        Appends the results of the Detectron tests for each strategy to self.test_results.

        Args:
            strategies (Union[Type[DetectronStrategy], List[Type[DetectronStrategy]]]): Class type or list of strategy class types.

        Returns:
            list: Updated list containing results for each strategy.
        """
        # Ensure strategies is a list of classes
        if isinstance(strategies, Type):
            strategies = [strategies]  # Convert single class type to list

        for strategy_class in strategies:
            if not issubclass(strategy_class, DetectronStrategy):
                raise TypeError("Each strategy must be a subclass of DetectronStrategy.")

            strategy_results = strategy_class.execute(self.cal_record, self.test_record)
            strategy_name = strategy_class.__name__
            strategy_results['Strategy'] = strategy_name
            self.test_results.append(strategy_results)

        return self.test_results

    def save(self, file_path: str, file_name: str = 'detectron_results'):
        """
        Saves the Detectron results to JSON format.

        Args:
            file_path (str): The file path where the results should be saved.
            file_name (str): The file name.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)
        with open(f'{file_path}/{file_name}.json', 'w') as file:
            json.dump(self.test_results, file, indent=4)
        
    
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
        
        # calculate the detectron test
        return detectron_results



