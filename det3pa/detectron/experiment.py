"""
This module include the execution of the detectron method and the handling of the results
"""
from __future__ import annotations
import json
from typing import Optional
from warnings import warn

from det3pa.detectron.record import DetectronRecordsManager
from det3pa.detectron.ensemble import DetectronEnsemble, BaseModelManager, DatasetsManager
from det3pa.detectron.strategies import *

from abc import ABC, abstractmethod

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
        self.test_results = None

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

    def get_experiments_results(self, strategy : DetectronStrategy, significance_level):
        """
        Executes the Detectron tests using the specified strategy and records.

        Args:
            strategy (DetectronStrategy): The strategy to use for detecting discrepancies.
            significance_level (float): The significance level for the statistical tests.

        Returns:
            dict: Results from executing the Detectron test.
        """        
        return(strategy.execute(self.cal_record, self.test_record, significance_level))
    
    def set_experiments_results(self, results : dict):
        """
        Stores the results of the Detectron tests.

        Args:
            results (dict): The results to store.
        """
        self.test_results = results

    def evaluate_detectron(self, strategy:DetectronStrategy, significance_level):
        """
        Evaluates the Detectron using the specified strategy.

        Args:
            strategy (DetectronStrategy): The strategy to evaluate.
            significance_level (float): The significance level for the evaluation.

        Returns:
            dict: Results from evaluating the Detectron.
        """
        return(strategy.evaluate(self.cal_record, self.test_record, significance_level))
    
    def save(self, file_path: str):
        """
        Saves the Detectron results to JSON format.

        Args:
            file_path (str): The file path where the results should be saved.
        """
        with open(f'{file_path}/detectron_results.json', 'w') as file:
            json.dump(self.test_results, file, indent=4)
        
    
class DetectronExperiment:
    """
    Abstract base class that defines the protocol for running Detectron experiments.

    Methods:
        run: Orchestrates the entire process of a Detectron experiment using specified parameters and strategies.
    """
    @abstractmethod
    def run(    datasets: DatasetsManager,
                training_params: dict,
                base_model_manager: BaseModelManager,
                samples_size : int = 20,
                detectron_result: DetectronResult = None,
                calib_result:DetectronRecordsManager=None,
                ensemble_size=10,
                num_calibration_runs=100,
                patience=3,
                significance_level=0.05, 
                test_strategy=DisagreementStrategy_z_mean,
                evaluate_detectron=False, 
                allow_margin : bool = False, 
                margin = 0.05):
        """
        Orchestrates the process of a Detectron experiment, including ensemble training and testing, and strategy evaluation.

        Args:
            datasets (DatasetsManager): Manages the datasets used in the experiment.
            training_params (dict): Parameters for training the cdcs within the ensembles.
            base_model_manager (BaseModelManager): Manager for the base model operations.
            samples_size (int): Number of samples to use in each Detectron run. Defaults to 20.
            detectron_result (Optional[DetectronResult]): Pre-existing results to bypass training phase. Defaults to None.
            calib_result (Optional[DetectronRecordsManager]): Calibration results, if provided. Defaults to None.
            ensemble_size (int): Number of models in each ensemble. Defaults to 10.
            num_calibration_runs (int): Number of calibration runs. Defaults to 100.
            patience (int): Number of iterations with no improvement before stopping. Defaults to 3.
            significance_level (float): Significance level for statistical tests. Defaults to 0.05.
            test_strategy (Any): Strategy for evaluating the Detectron results. Defaults to DisagreementStrategy_z_mean.
            evaluate_detectron (bool): Whether to perform an evaluation of the Detectron. Defaults to False.
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
        reference_set = datasets.get_reference_data(return_instance=True)
        if detectron_result is None:
            if reference_set is not None:
                test_size = len(reference_set)
                assert test_size > samples_size, \
                    "The reference set must be larger than the testing set to perform statistical bootstrapping"
                if test_size < 2 * samples_size:
                    warn("The reference set is smaller than twice the testing set, this may lead to poor calibration")
                if calib_result is not None:
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
                
                test_record = testing_ensemble.evaluate_ensemble(datasets=datasets, 
                                                                n_runs=num_calibration_runs, 
                                                                samples_size=samples_size, 
                                                                training_params=training_params,
                                                                set='testing', 
                                                                patience=patience,
                                                                allow_margin=allow_margin,
                                                                margin=margin)

        else:
            cal_record = detectron_result.cal_record
            test_record = detectron_result.test_record
            assert cal_record.sample_size == test_record.sample_size, \
                "The calibration record must have been generated with the same sample size as the observation set"
            
    
        # save the detectron runs results
        detectron_results = DetectronResult(cal_record, test_record)
        # calculate the detectron test
        experiment_results = detectron_results.get_experiments_results(test_strategy, significance_level)
        detectron_results.set_experiments_results(experiment_results)
        # evaluate the detectron if needed
        if evaluate_detectron:
            evaluation_results = detectron_results.evaluate_detectron(test_strategy, significance_level)
        else:
            evaluation_results = None
        return detectron_results, experiment_results, evaluation_results
    

