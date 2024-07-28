"""
Compares between two ``DetectronExperiment``.
"""
import json
import os
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from sklearn.model_selection import train_test_split

class DetectronComparison:
    """
    Class to compare the results of two Med3paExperiment instances.
    """
    def __init__(self, results1_path: str, results2_path: str) -> None:
        self.results1_path = os.path.abspath(results1_path)
        self.results2_path = os.path.abspath(results2_path)
        self.detectron_results_comparaison = {}
        self.config_file = {}
        self.model_evaluation_comparaison = {}
        self.rejection_counts_comparaison = {}
        self._check_experiment_name()

    def _check_experiment_name(self) -> None:
        """
        Checks if the experiment_name in the config_file of both results paths is the same.
        If not, raises a ValueError. Also sets the flag for Detectron comparison if applicable.
        """
        config_file_1 = os.path.join(self.results1_path, 'experiment_config.json')
        config_file_2 = os.path.join(self.results2_path, 'experiment_config.json')

        with open(config_file_1, 'r') as f1, open(config_file_2, 'r') as f2:
            config1 = json.load(f1)
            config2 = json.load(f2)

        if config1['experiment_name'] != "DetectronExperiment":
            raise ValueError("Only DetectronExperiment can be compared using this class")
        
        if config1['experiment_name'] != config2['experiment_name']:
            raise ValueError("The two results are not from the same experiment.")
        
    def compare_detectron_results(self):
        """
        Compares detectron results two sets of results and stores them in a dictionary.
        """
        combined = {}
        file_1 = os.path.join(self.results1_path, 'detectron_results.json')
        file_2 = os.path.join(self.results2_path, 'detectron_results.json')

        with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
            detectron1 = json.load(f1)
            detectron2 = json.load(f2)

        combined['detectron_results1'] = detectron1
        combined['detectron_results2'] = detectron2

        self.detectron_results_comparaison = combined

    def compare_evaluation(self):
        """
        Compares model evaluations two sets of results and stores them in a dictionary.
        """
        combined = {}
        file_1 = os.path.join(self.results1_path, 'model_evaluation.json')
        file_2 = os.path.join(self.results2_path, 'model_evaluation.json')

        with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
            detectron1 = json.load(f1)
            detectron2 = json.load(f2)

        combined['model_evaluation1'] = detectron1
        combined['model_evaluation2'] = detectron2

        self.model_evaluation_comparaison = combined

    def compare_counts(self):
        """
        Compares rejection two sets of results and stores them in a dictionary.
        """
        combined = {}
        file_1 = os.path.join(self.results1_path, 'rejection_counts.json')
        file_2 = os.path.join(self.results2_path, 'rejection_counts.json')

        with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
            detectron1 = json.load(f1)
            detectron2 = json.load(f2)

        combined['rejection_counts1'] = detectron1
        combined['rejection_counts2'] = detectron2

        self.rejection_counts_comparaison = combined

    def compare_config(self):
        """
        Compares the config files of the two experiments.
        """
        combined = {}
        config_file_1 = os.path.join(self.results1_path, 'experiment_config.json')
        config_file_2 = os.path.join(self.results2_path, 'experiment_config.json')

        with open(config_file_1, 'r') as f1, open(config_file_2, 'r') as f2:
            config1 = json.load(f1)
            config2 = json.load(f2)

        combined['datasets'] = {}
        
        if config1["datasets"] == config2["datasets"]:
            combined['datasets']['different'] = False
        else:
            combined['datasets']['different'] = True

        combined['datasets']['datasets1'] = config1["datasets"]
        combined['datasets']['datasets2'] = config2["datasets"]

        combined['base_model'] = {}

        if config1["base_model"] == config2["base_model"]:
            combined['base_model']['different'] = False
        else:
            combined['base_model']['different'] = True

        combined['base_model']['base_model1'] = config1["base_model"]
        combined['base_model']['base_model2'] = config2["base_model"]

        combined['detectron_params'] = {}

        if config1["detectron_params"] == config2["detectron_params"]:
            combined['detectron_params']['different'] = False
        else:
            combined['detectron_params']['different'] = True

        combined['detectron_params']['detectron_params1'] = config1["detectron_params"]
        combined['detectron_params']['detectron_params2'] = config2["detectron_params"]

        self.config_file = combined
        
    def compare_experiments(self):
        """
        Compares the experiments by detectron_results.
        """
        self.compare_detectron_results()
        self.compare_counts()
        self.compare_evaluation()
        self.compare_config()

    def save(self, directory_path: str) -> None:
        """
        Saves the comparison results to a specified directory.

        Args:
            directory_path (str): The directory where the comparison results will be saved.
        """
        # Ensure the main directory exists
        os.makedirs(directory_path, exist_ok=True)

        global_comparaison_path = os.path.join(directory_path, 'detectron_results_comparaison.json')
        with open(global_comparaison_path, 'w') as f:
                json.dump(self.detectron_results_comparaison, f, indent=4)
        
        config_path = os.path.join(directory_path, 'experiment_config_comparaison.json')
        with open(config_path, 'w') as f:
                json.dump(self.config_file, f, indent=4)

        eval_path = os.path.join(directory_path, 'model_evaluation_comparaison.json')
        with open(eval_path, 'w') as f:
                json.dump(self.model_evaluation_comparaison, f, indent=4)

        counts_path = os.path.join(directory_path, 'rejection_counts_comparaison.json')
        with open(counts_path, 'w') as f:
                json.dump(self.rejection_counts_comparaison, f, indent=4)
